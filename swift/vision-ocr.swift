import AppKit
import CoreGraphics
import Foundation
import ImageIO
import PDFKit
import UniformTypeIdentifiers
import Vision

struct NormalizedBBox: Codable {
	let x: Double
	let y: Double
	let width: Double
	let height: Double
}

struct OcrLine: Codable {
	let text: String
	let confidence: Double
	let bbox: NormalizedBBox
}

struct OcrPageResult: Codable {
	let page: Int
	let widthPx: Int
	let heightPx: Int
	let lines: [OcrLine]
	let textLayerLines: [OcrLine]?
	let figureCandidates: [NormalizedBBox]?
}

struct OcrBatchOutput: Codable {
	let totalPages: Int
	let pages: [OcrPageResult]
}

enum HelperError: Error, CustomStringConvertible {
	case message(String)

	var description: String {
		switch self {
		case .message(let msg):
			return msg
		}
	}
}

func clamp01(_ value: Double) -> Double {
	return min(1.0, max(0.0, value))
}

func renderPdfPageToCGImage(page: PDFPage, dpi: Double) throws -> CGImage {
	let pageRect = page.bounds(for: .mediaBox)
	let scale = max(1.0, dpi / 72.0)
	let width = max(1, Int((pageRect.width * scale).rounded()))
	let height = max(1, Int((pageRect.height * scale).rounded()))

	let colorSpace = CGColorSpaceCreateDeviceRGB()
	let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
	guard
		let context = CGContext(
			data: nil,
			width: width,
			height: height,
			bitsPerComponent: 8,
			bytesPerRow: 0,
			space: colorSpace,
			bitmapInfo: bitmapInfo
		)
	else {
		throw HelperError.message("Failed to allocate CGContext for PDF page rendering")
	}

	context.setFillColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
	context.fill(CGRect(x: 0, y: 0, width: width, height: height))

	context.saveGState()
	context.scaleBy(x: scale, y: scale)
	page.draw(with: .mediaBox, to: context)
	context.restoreGState()

	guard let cgImage = context.makeImage() else {
		throw HelperError.message("Failed to create CGImage from rendered PDF page")
	}
	return cgImage
}

func textLayerLines(page: PDFPage) -> [OcrLine]? {
	let pageRect = page.bounds(for: .mediaBox)
	// Do not correlate text-layer coordinates with the raster when their transforms differ.
	guard page.rotation % 360 == 0, pageRect.minX == 0, pageRect.minY == 0,
		pageRect.width > 0, pageRect.height > 0,
		let text = page.string, !text.isEmpty,
		let selection = page.selection(for: NSRange(location: 0, length: (text as NSString).length))
	else { return nil }

	let lines = selection.selectionsByLine().compactMap { selectedLine -> OcrLine? in
		guard let value = selectedLine.string?.trimmingCharacters(in: .whitespacesAndNewlines),
			!value.isEmpty else { return nil }
		let rect = selectedLine.bounds(for: page)
		guard rect.width > 0, rect.height > 0 else { return nil }
		return OcrLine(
			text: value,
			confidence: 0.8,
			bbox: NormalizedBBox(
				x: clamp01(Double(rect.minX / pageRect.width)),
				y: clamp01(1.0 - Double(rect.maxY / pageRect.height)),
				width: clamp01(Double(rect.width / pageRect.width)),
				height: clamp01(Double(rect.height / pageRect.height))
			)
		)
	}
	return lines.isEmpty ? nil : lines
}

/// Marks each ink pixel that sits in a horizontal or vertical run at least `minimumLength` long.
func longStraightRuns(ink: [Bool], width: Int, height: Int, minimumLength: Int) -> [Bool] {
	var marked = [Bool](repeating: false, count: width * height)
	for y in 0..<height {
		var runStart = -1
		for x in 0...width {
			let isInk = x < width && ink[y * width + x]
			if isInk {
				if runStart < 0 {
					runStart = x
				}
			} else if runStart >= 0 {
				if x - runStart >= minimumLength {
					for k in runStart..<x {
						marked[y * width + k] = true
					}
				}
				runStart = -1
			}
		}
	}
	for x in 0..<width {
		var runStart = -1
		for y in 0...height {
			let isInk = y < height && ink[y * width + x]
			if isInk {
				if runStart < 0 {
					runStart = y
				}
			} else if runStart >= 0 {
				if y - runStart >= minimumLength {
					for k in runStart..<y {
						marked[k * width + x] = true
					}
				}
				runStart = -1
			}
		}
	}
	return marked
}

/// A piece of recognized text that the figure search erases before it groups ink.
struct TextBox {
	let bbox: NormalizedBBox
	/// A short phrase, such as a node value, a variable, or a box title. Only labels can be
	/// restored into a drawing; sentences stay prose even when they sit beside one.
	let isLabel: Bool
	/// Text with a word in it, such as "Machine" or "30". A lone glyph does not count: a circle
	/// in a board is often read as "0", and a board is not a table.
	let isWord: Bool
}

func hasLetterOrDigit(_ text: String) -> Bool {
	return text.contains { $0.isLetter || $0.isNumber }
}

/// True for a line made only of code punctuation, such as a lone closing brace.
func isCodePunctuation(_ text: String) -> Bool {
	let codeCharacters = Set("{}[]();")
	let significant = text.filter { !$0.isWhitespace }
	return !significant.isEmpty && significant.allSatisfy { codeCharacters.contains($0) }
}

/// True when some word in the text has two or more letters or digits.
func isWordText(_ text: String) -> Bool {
	return text.split(whereSeparator: { $0.isWhitespace }).contains { token in
		token.filter { $0.isLetter || $0.isNumber }.count >= 2
	}
}

/// A label is a short phrase without sentence punctuation. A sentence, such as a caption that
/// ends in a period, is prose even when it sits beside a drawing, and so is a code line. A line
/// that ends in a colon introduces the drawing, so it is prose too.
func isLabelText(_ text: String) -> Bool {
	let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
	let words = trimmed.split(whereSeparator: { $0.isWhitespace }).filter { hasLetterOrDigit(String($0)) }
	return (1...8).contains(words.count)
		&& !trimmed.hasSuffix(".")
		&& !trimmed.hasSuffix(":")
		&& !trimmed.contains { ";{}".contains($0) }
}

/// Finds drawings on a page: dark ink that Vision did not recognize as text, grouped into
/// connected regions. Recognized text is erased first, except short labels that touch a drawing,
/// so each region holds its outlines, arrows, and the labels that belong to them. The NLP stage
/// claims those labels from the layout.
/// Regions with many long horizontal rules and words inside are tables, which the text pipeline
/// handles. A board with lone glyphs in its cells is still a drawing.
func detectFigureCandidates(cgImage: CGImage, textBoxes: [TextBox]) -> [NormalizedBBox] {
	let analysisScale = 2
	let inkThreshold: UInt8 = 200
	let mergeRadiusPx = 8
	let labelReachPx = 14
	let structuralRunPx = 24
	// Lines wider than this are running text. Labels are narrower.
	let runningTextWidthFraction = 0.4
	// Drawing ink is ink that survives text erasure. Restored labels are wanted in a drawing, but
	// they cannot make one: glyph fragments left beside code would otherwise pass as a drawing.
	let minimumInkPixels = 250
	let minimumWidthFraction = 0.12
	let minimumHeightFraction = 0.04
	// A row of small boxes, such as a linked list, can be under 2% of a page tall. Only a drawing
	// with no full-width rule may be that short, so a page header or footer rule stays out.
	let minimumRowHeightFraction = 0.015
	// A horizontal run this long, relative to its drawing, is a rule. Anti-aliasing and table
	// cell borders keep rules from reaching the full width.
	let fullWidthRunFraction = 0.6
	let maximumFullWidthRuns = 2

	let width = max(1, cgImage.width / analysisScale)
	let height = max(1, cgImage.height / analysisScale)
	let pixelCount = width * height

	var gray = [UInt8](repeating: 255, count: pixelCount)
	let drawn = gray.withUnsafeMutableBytes { buffer -> Bool in
		guard
			let context = CGContext(
				data: buffer.baseAddress,
				width: width,
				height: height,
				bitsPerComponent: 8,
				bytesPerRow: width,
				space: CGColorSpaceCreateDeviceGray(),
				bitmapInfo: CGImageAlphaInfo.none.rawValue
			)
		else {
			return false
		}
		context.interpolationQuality = .high
		context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
		return true
	}
	guard drawn else {
		return []
	}

	let original = gray.map { $0 < inkThreshold }
	// Box edges and table rules are long straight runs that glyph strokes never form. They stay
	// ink even inside a text box, so a label that sits on a border cannot cut the outline apart.
	let structural = longStraightRuns(
		ink: original, width: width, height: height, minimumLength: structuralRunPx)
	func textRect(_ box: NormalizedBBox) -> (left: Int, top: Int, right: Int, bottom: Int)? {
		let left = max(0, Int(box.x * Double(width)))
		let right = min(width - 1, Int(ceil((box.x + box.width) * Double(width))))
		let top = max(0, Int(box.y * Double(height)))
		let bottom = min(height - 1, Int(ceil((box.y + box.height) * Double(height))))
		guard left <= right, top <= bottom else {
			return nil
		}
		return (left, top, right, bottom)
	}
	var ink = original
	for box in textBoxes {
		guard let rect = textRect(box.bbox) else {
			continue
		}
		for y in rect.top...rect.bottom {
			for x in rect.left...rect.right where !structural[y * width + x] {
				ink[y * width + x] = false
			}
		}
	}

	// Summed-area tables count the ink in any rectangle in constant time.
	let rowStride = width + 1
	func summedAreaTable(_ mask: [Bool]) -> [Int32] {
		var table = [Int32](repeating: 0, count: rowStride * (height + 1))
		for y in 0..<height {
			var rowTotal: Int32 = 0
			for x in 0..<width {
				if mask[y * width + x] {
					rowTotal += 1
				}
				table[(y + 1) * rowStride + (x + 1)] = table[y * rowStride + (x + 1)] + rowTotal
			}
		}
		return table
	}
	func inkCount(_ table: [Int32], left: Int, top: Int, right: Int, bottom: Int) -> Int32 {
		return table[(bottom + 1) * rowStride + (right + 1)]
			- table[top * rowStride + (right + 1)]
			- table[(bottom + 1) * rowStride + left]
			+ table[top * rowStride + left]
	}

	// Ink that survives erasure is the drawing's own. Restored labels add to a drawing but cannot
	// make one: glyph fragments left beside text would otherwise pass for a drawing.
	let surviving = ink
	// A label that touches a drawing belongs to it: a flowchart arrow points at its text, and a
	// tree's branches end at its labels. A sentence beside a drawing is prose and stays erased.
	let erasedTable = summedAreaTable(ink)
	var restoredAny = false
	for box in textBoxes where box.isLabel && box.bbox.width < runningTextWidthFraction {
		guard let rect = textRect(box.bbox) else {
			continue
		}
		let beside =
			inkCount(
				erasedTable,
				left: max(0, rect.left - labelReachPx),
				top: max(0, rect.top - labelReachPx),
				right: min(width - 1, rect.right + labelReachPx),
				bottom: min(height - 1, rect.bottom + labelReachPx))
			- inkCount(
				erasedTable, left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom)
		if beside > 0 {
			restoredAny = true
			for y in rect.top...rect.bottom {
				for x in rect.left...rect.right {
					ink[y * width + x] = original[y * width + x]
				}
			}
		}
	}

	// Dilate the ink by mergeRadiusPx, so strokes that are a few pixels apart are grouped into
	// one drawing.
	let integral = restoredAny ? summedAreaTable(ink) : erasedTable
	var merged = [Bool](repeating: false, count: pixelCount)
	for y in 0..<height {
		let top = max(0, y - mergeRadiusPx)
		let bottom = min(height - 1, y + mergeRadiusPx)
		for x in 0..<width {
			let left = max(0, x - mergeRadiusPx)
			let right = min(width - 1, x + mergeRadiusPx)
			merged[y * width + x] =
				inkCount(integral, left: left, top: top, right: right, bottom: bottom) > 0
		}
	}

	var labels = [Int32](repeating: -1, count: pixelCount)
	var pending: [Int] = []
	func visit(_ next: Int, label: Int32) {
		guard merged[next], labels[next] < 0 else {
			return
		}
		labels[next] = label
		pending.append(next)
	}
	var labelCount = 0
	for start in 0..<pixelCount where merged[start] && labels[start] < 0 {
		let label = Int32(labelCount)
		visit(start, label: label)
		while let index = pending.popLast() {
			let x = index % width
			let y = index / width
			if x > 0 {
				visit(index - 1, label: label)
			}
			if x + 1 < width {
				visit(index + 1, label: label)
			}
			if y > 0 {
				visit(index - width, label: label)
			}
			if y + 1 < height {
				visit(index + width, label: label)
			}
		}
		labelCount += 1
	}

	struct ComponentStats {
		var minX = Int.max
		var maxX = -1
		var minY = Int.max
		var maxY = -1
		var drawingInk = 0
		var ruleRuns = 0
	}
	var stats = [ComponentStats](repeating: ComponentStats(), count: labelCount)
	for y in 0..<height {
		for x in 0..<width where ink[y * width + x] {
			let label = Int(labels[y * width + x])
			guard label >= 0 else {
				continue
			}
			stats[label].minX = min(stats[label].minX, x)
			stats[label].maxX = max(stats[label].maxX, x)
			stats[label].minY = min(stats[label].minY, y)
			stats[label].maxY = max(stats[label].maxY, y)
			if surviving[y * width + x] {
				stats[label].drawingInk += 1
			}
		}
	}
	// A rule that spans its whole drawing is a box edge or a table rule. A box has two of them.
	// A table has three or more. A rule that is several pixels thick spans several rows, so
	// consecutive rows count as one rule.
	var lastRuleRow = [Int](repeating: -2, count: labelCount)
	for y in 0..<height {
		var runStart: Int?
		for x in 0...width {
			let isInk = x < width && ink[y * width + x]
			if isInk {
				if runStart == nil {
					runStart = x
				}
			} else if let start = runStart {
				let label = Int(labels[y * width + start])
				if label >= 0 {
					let componentWidth = stats[label].maxX - stats[label].minX + 1
					if Double(x - start) >= fullWidthRunFraction * Double(componentWidth) {
						if lastRuleRow[label] != y - 1 {
							stats[label].ruleRuns += 1
						}
						lastRuleRow[label] = y
					}
				}
				runStart = nil
			}
		}
	}

	var candidates: [NormalizedBBox] = []
	for entry in stats where entry.drawingInk >= minimumInkPixels {
		let boxWidth = entry.maxX - entry.minX + 1
		let boxHeight = entry.maxY - entry.minY + 1
		let requiredHeight = entry.ruleRuns == 0 ? minimumRowHeightFraction : minimumHeightFraction
		// Table cells hold words, which erasure leaves as gaps inside the rules. A grid drawing,
		// such as a board, holds at most lone glyphs, so it stays a drawing.
		let holdsText = textBoxes.contains { box in
			let centerX = (box.bbox.x + box.bbox.width / 2) * Double(width)
			let centerY = (box.bbox.y + box.bbox.height / 2) * Double(height)
			return box.isWord
				&& centerX >= Double(entry.minX) && centerX <= Double(entry.maxX)
				&& centerY >= Double(entry.minY) && centerY <= Double(entry.maxY)
		}
		let isTable = entry.ruleRuns > maximumFullWidthRuns && holdsText
		guard
			Double(boxWidth) >= minimumWidthFraction * Double(width),
			Double(boxHeight) >= requiredHeight * Double(height),
			!isTable
		else {
			continue
		}
		candidates.append(
			NormalizedBBox(
				x: Double(entry.minX) / Double(width),
				y: Double(entry.minY) / Double(height),
				width: Double(boxWidth) / Double(width),
				height: Double(boxHeight) / Double(height)
			))
	}
	return candidates.sorted { $0.y < $1.y }
}

func recognizePage(pageNumber: Int, cgImage: CGImage, language: String, pdfPage: PDFPage) throws -> OcrPageResult {
	var recognizedLines: [OcrLine] = []
	var requestError: Error?

	let request = VNRecognizeTextRequest { req, err in
		if let err = err {
			requestError = err
			return
		}
		guard let observations = req.results as? [VNRecognizedTextObservation] else {
			return
		}
		for obs in observations {
			guard let candidate = obs.topCandidates(1).first else {
				continue
			}
			let rawText = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
			if rawText.isEmpty {
				continue
			}
			let rect = obs.boundingBox
			let x = clamp01(Double(rect.origin.x))
			let y = clamp01(1.0 - Double(rect.origin.y) - Double(rect.size.height))
			let w = clamp01(Double(rect.size.width))
			let h = clamp01(Double(rect.size.height))

			recognizedLines.append(
				OcrLine(
					text: rawText,
					confidence: clamp01(Double(candidate.confidence)),
					bbox: NormalizedBBox(x: x, y: y, width: w, height: h)
				)
			)
		}
	}

	request.recognitionLevel = .accurate
	request.usesLanguageCorrection = true
	if !language.isEmpty {
		request.recognitionLanguages = [language]
	}

	let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
	try handler.perform([request])

	if let err = requestError {
		throw err
	}

	recognizedLines.sort { a, b in
		let yDiff = a.bbox.y - b.bbox.y
		let avgHeight = max(0.008, (a.bbox.height + b.bbox.height) * 0.5)
		if abs(yDiff) > avgHeight * 0.55 {
			return a.bbox.y < b.bbox.y
		}
		return a.bbox.x < b.bbox.x
	}

	let layerLines = textLayerLines(page: pdfPage)
	// Vision reads arrows such as "→" and bracket shapes in drawings as symbols. Erasing those
	// would cut connectors, so Vision text is erased only when it has a letter or digit. The PDF
	// text layer is exact, so it also erases code punctuation, such as the lone `}` that closes a
	// code line. Left in place, that brace would start a drawing around the code.
	let visionBoxes = recognizedLines
		.filter { hasLetterOrDigit($0.text) }
		.map { TextBox(bbox: $0.bbox, isLabel: isLabelText($0.text), isWord: isWordText($0.text)) }
	let layerBoxes = (layerLines ?? [])
		.filter { hasLetterOrDigit($0.text) || isCodePunctuation($0.text) }
		.map { TextBox(bbox: $0.bbox, isLabel: isLabelText($0.text), isWord: isWordText($0.text)) }

	return OcrPageResult(
		page: pageNumber,
		widthPx: cgImage.width,
		heightPx: cgImage.height,
		lines: recognizedLines,
		textLayerLines: layerLines,
		figureCandidates: detectFigureCandidates(
			cgImage: cgImage, textBoxes: visionBoxes + layerBoxes)
	)
}

func parsePageList(_ raw: String, totalPages: Int) throws -> [Int] {
	if raw == "all" || raw.isEmpty {
		return Array(1...max(1, totalPages))
	}
	var pages = Set<Int>()
	let parts = raw.split(separator: ",")
	for partSub in parts {
		let part = partSub.trimmingCharacters(in: .whitespaces)
		if part.contains("-") {
			let bounds = part.split(separator: "-")
			guard bounds.count == 2,
				let start = Int(bounds[0]),
				let end = Int(bounds[1]),
				start >= 1,
				end >= start
			else {
				throw HelperError.message("Invalid page range: \(part)")
			}
			let clampedEnd = min(totalPages, end)
			if start <= clampedEnd {
				for p in start...clampedEnd {
					pages.insert(p)
				}
			}
		} else {
			guard let p = Int(part), p >= 1 else {
				throw HelperError.message("Invalid page number: \(part)")
			}
			if p <= totalPages {
				pages.insert(p)
			}
		}
	}
	return pages.sorted()
}

func writePng(cgImage: CGImage, to outputPath: String) throws {
	let url = URL(fileURLWithPath: outputPath)
	let dir = url.deletingLastPathComponent()
	try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

	guard
		let destination = CGImageDestinationCreateWithURL(
			url as CFURL,
			UTType.png.identifier as CFString,
			1,
			nil
		)
	else {
		throw HelperError.message("Failed to create PNG image destination at \(outputPath)")
	}
	CGImageDestinationAddImage(destination, cgImage, nil)
	if !CGImageDestinationFinalize(destination) {
		throw HelperError.message("Failed to finalize PNG write at \(outputPath)")
	}
}

func cropCGImage(_ image: CGImage, cropSpec: String, addPadding: Bool = true) throws -> CGImage {
	let parts = cropSpec.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
	guard parts.count == 4 else {
		throw HelperError.message("Invalid crop spec: \(cropSpec). Expected x,y,width,height")
	}
	let imgW = Double(image.width)
	let imgH = Double(image.height)
	let padY = addPadding ? 0.012 : 0.0
	let padX = addPadding ? 0.01 : 0.0
	let x = clamp01(parts[0] - padX) * imgW
	let y = clamp01(parts[1] - padY) * imgH
	let w = min(imgW - x, clamp01(parts[2] + padX * 2.0) * imgW)
	let h = min(imgH - y, clamp01(parts[3] + padY * 2.0) * imgH)

	let rect = CGRect(x: x.rounded(.down), y: y.rounded(.down), width: max(1.0, w.rounded(.up)), height: max(1.0, h.rounded(.up)))
	guard let cropped = image.cropping(to: rect) else {
		throw HelperError.message("Failed to crop CGImage with rect \(rect)")
	}
	return cropped
}

func trimWhitespaceAndPad(_ image: CGImage) -> CGImage {
	let bytesPerPixel = 4
	let bytesPerRow = image.width * bytesPerPixel
	var pixels = [UInt8](repeating: 255, count: bytesPerRow * image.height)
	let colorSpace = CGColorSpaceCreateDeviceRGB()
	let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

	pixels.withUnsafeMutableBytes { rawBuffer in
		guard
			let data = rawBuffer.baseAddress,
			let context = CGContext(
				data: data,
				width: image.width,
				height: image.height,
				bitsPerComponent: 8,
				bytesPerRow: bytesPerRow,
				space: colorSpace,
				bitmapInfo: bitmapInfo
			)
		else {
			return
		}
		context.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
		context.fill(CGRect(x: 0, y: 0, width: image.width, height: image.height))
		context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
	}

	var minX = image.width
	var minY = image.height
	var maxX = -1
	var maxY = -1
	for y in 0..<image.height {
		for x in 0..<image.width {
			let pixelOffset = y * bytesPerRow + x * bytesPerPixel
			let red = Double(pixels[pixelOffset])
			let green = Double(pixels[pixelOffset + 1])
			let blue = Double(pixels[pixelOffset + 2])
			let luminance = (299 * red + 587 * green + 114 * blue) / 1000
			if luminance < 245 {
				minX = min(minX, x)
				minY = min(minY, y)
				maxX = max(maxX, x)
				maxY = max(maxY, y)
			}
		}
	}

	guard maxX >= minX, maxY >= minY else {
		return image
	}

	// A small margin keeps anti-aliased edges. A wide one would take in the caption or the
	// paragraph beside the drawing, which is prose, not part of the figure.
	let targetPadding = 6
	let availablePadding = min(minX, minY, image.width - maxX - 1, image.height - maxY - 1)
	let padding = max(0, min(targetPadding, availablePadding))
	let cropX = max(0, minX - padding)
	let cropY = max(0, minY - padding)
	let cropRight = min(image.width, maxX + padding + 1)
	let cropBottom = min(image.height, maxY + padding + 1)
	let rect = CGRect(
		x: CGFloat(cropX),
		y: CGFloat(cropY),
		width: CGFloat(cropRight - cropX),
		height: CGFloat(cropBottom - cropY)
	)
	return image.cropping(to: rect) ?? image
}

func run() throws {
	let args = Array(CommandLine.arguments.dropFirst())
	guard let mode = args.first else {
		throw HelperError.message("Usage: vision-ocr <ocr|render-page|info> [options]")
	}

	var pdfPath = ""
	var pagesArg = "all"
	var pageArg = 1
	var langArg = "en-US"
	var dpiArg = 220.0
	var outPath = ""
	var cropArg = ""
	var trimWhitespace = false

	var i = 1
	while i < args.count {
		let flag = args[i]
		switch flag {
		case "--pdf":
			i += 1
			if i < args.count { pdfPath = args[i] }
		case "--pages":
			i += 1
			if i < args.count { pagesArg = args[i] }
		case "--page":
			i += 1
			if i < args.count, let val = Int(args[i]) { pageArg = val }
		case "--lang":
			i += 1
			if i < args.count { langArg = args[i] }
		case "--dpi":
			i += 1
			if i < args.count, let val = Double(args[i]) { dpiArg = val }
		case "--out":
			i += 1
			if i < args.count { outPath = args[i] }
		case "--crop":
			i += 1
			if i < args.count { cropArg = args[i] }
		case "--trim-whitespace":
			trimWhitespace = true
		default:
			throw HelperError.message("Unknown argument: \(flag)")
		}
		i += 1
	}

	guard !pdfPath.isEmpty else {
		throw HelperError.message("Missing required --pdf <path> argument")
	}

	let pdfUrl = URL(fileURLWithPath: pdfPath)
	guard let doc = PDFDocument(url: pdfUrl) else {
		throw HelperError.message("Could not open PDF at path: \(pdfPath)")
	}

	let totalPages = doc.pageCount

	switch mode {
	case "info":
		let output = OcrBatchOutput(totalPages: totalPages, pages: [])
		let data = try JSONEncoder().encode(output)
		FileHandle.standardOutput.write(data)
		FileHandle.standardOutput.write(Data("\n".utf8))

	case "ocr":
		let targetPages = try parsePageList(pagesArg, totalPages: totalPages)
		var results: [OcrPageResult] = []
		results.reserveCapacity(targetPages.count)

		for p in targetPages {
			autoreleasepool {
				guard let pdfPage = doc.page(at: p - 1) else { return }
				do {
					let cgImage = try renderPdfPageToCGImage(page: pdfPage, dpi: dpiArg)
					let pageResult = try recognizePage(pageNumber: p, cgImage: cgImage, language: langArg, pdfPage: pdfPage)
					results.append(pageResult)
				} catch {
					fputs("Error on page \(p): \(error)\n", stderr)
				}
			}
		}

		let batch = OcrBatchOutput(totalPages: totalPages, pages: results)
		let data = try JSONEncoder().encode(batch)
		FileHandle.standardOutput.write(data)
		FileHandle.standardOutput.write(Data("\n".utf8))

	case "render-page":
		guard !outPath.isEmpty else {
			throw HelperError.message("Missing --out <path> for render-page")
		}
		guard pageArg >= 1 && pageArg <= totalPages, let pdfPage = doc.page(at: pageArg - 1) else {
			throw HelperError.message("Page \(pageArg) out of bounds (1..\(totalPages))")
		}
		let cgImage = try renderPdfPageToCGImage(page: pdfPage, dpi: dpiArg)
		let croppedImage = cropArg.isEmpty
			? cgImage
			: try cropCGImage(cgImage, cropSpec: cropArg, addPadding: !trimWhitespace)
		let finalImage = trimWhitespace ? trimWhitespaceAndPad(croppedImage) : croppedImage
		try writePng(cgImage: finalImage, to: outPath)

	default:
		throw HelperError.message("Unknown mode: \(mode)")
	}
}

do {
	try run()
} catch {
	fputs("vision-ocr error: \(error)\n", stderr)
	exit(1)
}
