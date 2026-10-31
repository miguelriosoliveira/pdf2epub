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

	return OcrPageResult(
		page: pageNumber,
		widthPx: cgImage.width,
		heightPx: cgImage.height,
		lines: recognizedLines,
		textLayerLines: textLayerLines(page: pdfPage)
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

	let contentWidth = maxX - minX + 1
	let contentHeight = maxY - minY + 1
	let targetPadding = max(4, Int((Double(max(contentWidth, contentHeight)) * 0.05).rounded()))
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
