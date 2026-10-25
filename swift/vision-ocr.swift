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

func recognizePage(pageNumber: Int, cgImage: CGImage, language: String) throws -> OcrPageResult {
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
		lines: recognizedLines
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

func cropCGImage(_ image: CGImage, cropSpec: String) throws -> CGImage {
	let parts = cropSpec.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
	guard parts.count == 4 else {
		throw HelperError.message("Invalid crop spec: \(cropSpec). Expected x,y,width,height")
	}
	let imgW = Double(image.width)
	let imgH = Double(image.height)
	let padY = 0.012
	let padX = 0.01
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
					let pageResult = try recognizePage(pageNumber: p, cgImage: cgImage, language: langArg)
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
		let finalImage = cropArg.isEmpty ? cgImage : try cropCGImage(cgImage, cropSpec: cropArg)
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
