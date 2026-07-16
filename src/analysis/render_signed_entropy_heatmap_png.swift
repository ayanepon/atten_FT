import AppKit
import Foundation

struct Row {
    let comparison: String
    let layer: Int
    let head: Int
    let delta: Double
    let significant: Bool
}

func parseCSV(_ path: String) throws -> [Row] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    let lines = text.split(whereSeparator: \.isNewline).map(String.init)
    guard let header = lines.first else { return [] }
    let names = header.split(separator: ",").map(String.init)
    func idx(_ name: String) -> Int {
        guard let i = names.firstIndex(of: name) else {
            fatalError("Missing CSV column: \(name)")
        }
        return i
    }
    let comparisonI = idx("comparison")
    let metricI = idx("metric")
    let layerI = idx("layer")
    let headI = idx("head")
    let deltaI = idx("cliffs_delta")
    let sigI = idx("significant_bh_0_05")

    var rows: [Row] = []
    for line in lines.dropFirst() {
        let cols = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        guard cols.count > sigI, cols[metricI] == "entropy_delta" else { continue }
        rows.append(Row(
            comparison: cols[comparisonI],
            layer: Int(cols[layerI]) ?? 0,
            head: Int(cols[headI]) ?? 0,
            delta: Double(cols[deltaI]) ?? 0.0,
            significant: cols[sigI] == "True"
        ))
    }
    return rows
}

func colorForDelta(_ value: Double) -> NSColor {
    let v = max(-1.0, min(1.0, value))
    let startBlue = (49.0, 82.0, 190.0)
    let white = (245.0, 245.0, 245.0)
    let endRed = (190.0, 20.0, 60.0)
    let rgb: (Double, Double, Double)
    if v < 0 {
        let t = v + 1.0
        rgb = (
            startBlue.0 + (white.0 - startBlue.0) * t,
            startBlue.1 + (white.1 - startBlue.1) * t,
            startBlue.2 + (white.2 - startBlue.2) * t
        )
    } else {
        let t = v
        rgb = (
            white.0 + (endRed.0 - white.0) * t,
            white.1 + (endRed.1 - white.1) * t,
            white.2 + (endRed.2 - white.2) * t
        )
    }
    return NSColor(calibratedRed: rgb.0 / 255.0, green: rgb.1 / 255.0, blue: rgb.2 / 255.0, alpha: 1.0)
}

func drawText(_ text: String, x: CGFloat, y: CGFloat, size: CGFloat, align: NSTextAlignment = .center, bold: Bool = false) {
    let font = bold ? NSFont.boldSystemFont(ofSize: size) : NSFont.systemFont(ofSize: size)
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: NSColor.black
    ]
    let attributed = NSAttributedString(string: text, attributes: attrs)
    let s = attributed.size()
    var drawX = x
    if align == .center {
        drawX -= s.width / 2.0
    } else if align == .right {
        drawX -= s.width
    }
    attributed.draw(at: NSPoint(x: drawX, y: y - s.height / 2.0))
}

func render(rows: [Row], comparison: String, output: String) throws {
    let width = 1100
    let height = 440
    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: width,
        pixelsHigh: height,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        fatalError("Failed to allocate bitmap")
    }
    let graphicsContext = NSGraphicsContext(bitmapImageRep: bitmap)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = graphicsContext
    defer { NSGraphicsContext.restoreGraphicsState() }

    NSColor.white.setFill()
    NSBezierPath(rect: NSRect(x: 0, y: 0, width: width, height: height)).fill()

    let nLayers = 16
    let nHeads = 8
    let left: CGFloat = 92
    let topFromTop: CGFloat = 28
    let cellW: CGFloat = 78
    let cellH: CGFloat = 20
    let gridW = CGFloat(nHeads) * cellW
    let gridH = CGFloat(nLayers) * cellH
    let top = CGFloat(height) - topFromTop - gridH

    var byKey: [String: Row] = [:]
    for r in rows where r.comparison == comparison {
        byKey["\(r.layer)-\(r.head)"] = r
    }

    for layer in 0..<nLayers {
        let y = top + CGFloat(layer) * cellH
        for head in 0..<nHeads {
            guard let r = byKey["\(layer)-\(head)"] else { continue }
            let x = left + CGFloat(head) * cellW
            colorForDelta(r.delta).setFill()
            NSBezierPath(rect: NSRect(x: x, y: y, width: cellW, height: cellH)).fill()
            if r.significant {
                drawText("*", x: x + cellW / 2.0, y: y + cellH * 0.50, size: 22, bold: true)
            }
        }
    }

    NSColor.black.setStroke()
    let border = NSBezierPath(rect: NSRect(x: left, y: top, width: gridW, height: gridH))
    border.lineWidth = 4
    border.stroke()

    for head in 0..<nHeads {
        let x = left + CGFloat(head) * cellW + cellW / 2.0
        drawText("\(head)", x: x, y: top - 24, size: 22)
    }
    for layer in 0..<nLayers {
        let y = top + CGFloat(layer) * cellH + cellH / 2.0
        drawText("\(layer)", x: left - 18, y: y, size: 18, align: .right)
    }
    drawText("Head", x: left + gridW / 2.0, y: top - 54, size: 28)

    let layerAttrs: [NSAttributedString.Key: Any] = [
        .font: NSFont.systemFont(ofSize: 28),
        .foregroundColor: NSColor.black
    ]
    let layerText = NSAttributedString(string: "Layer", attributes: layerAttrs)
    NSGraphicsContext.current?.cgContext.saveGState()
    NSGraphicsContext.current?.cgContext.translateBy(x: 30, y: top + gridH / 2.0)
    NSGraphicsContext.current?.cgContext.rotate(by: CGFloat.pi / 2.0)
    layerText.draw(at: NSPoint(x: -layerText.size().width / 2.0, y: -layerText.size().height / 2.0))
    NSGraphicsContext.current?.cgContext.restoreGState()

    let cbX = left + gridW + 38
    let cbY = top
    let cbW: CGFloat = 32
    let cbH = gridH
    let steps = 160
    for i in 0..<steps {
        let v = 1.0 - 2.0 * Double(i) / Double(steps - 1)
        let y = cbY + CGFloat(steps - 1 - i) * cbH / CGFloat(steps)
        colorForDelta(v).setFill()
        NSBezierPath(rect: NSRect(x: cbX, y: y, width: cbW, height: cbH / CGFloat(steps) + 1)).fill()
    }
    let cbBorder = NSBezierPath(rect: NSRect(x: cbX, y: cbY, width: cbW, height: cbH))
    cbBorder.lineWidth = 3
    cbBorder.stroke()
    for val in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let y = cbY + CGFloat((val + 1.0) / 2.0) * cbH
        let tick = NSBezierPath()
        tick.move(to: NSPoint(x: cbX + cbW, y: y))
        tick.line(to: NSPoint(x: cbX + cbW + 12, y: y))
        tick.lineWidth = 3
        tick.stroke()
        let label = val == floor(val) ? String(format: "%.0f", val) : String(format: "%.1f", val)
        drawText(label.replacingOccurrences(of: "-", with: "−"), x: cbX + cbW + 20, y: y, size: 17, align: .left)
    }

    let other = comparison == "ft_vs_pt" ? "PT" : "Unseen"
    let label = NSAttributedString(
        string: "Cliff's delta (FT - \(other))",
        attributes: [.font: NSFont.systemFont(ofSize: 18), .foregroundColor: NSColor.black]
    )
    NSGraphicsContext.current?.cgContext.saveGState()
    NSGraphicsContext.current?.cgContext.translateBy(x: cbX + cbW + 92, y: cbY + cbH / 2.0)
    NSGraphicsContext.current?.cgContext.rotate(by: CGFloat.pi / 2.0)
    label.draw(at: NSPoint(x: -label.size().width / 2.0, y: -label.size().height / 2.0))
    NSGraphicsContext.current?.cgContext.restoreGState()

    guard let png = bitmap.representation(using: .png, properties: [:]) else {
        fatalError("Failed to encode PNG")
    }
    try png.write(to: URL(fileURLWithPath: output))
}

let args = CommandLine.arguments
guard args.count == 3 else {
    fatalError("Usage: swift render_signed_entropy_heatmap_png.swift input.csv output_dir")
}
let input = args[1]
let outputDir = args[2]
let rows = try parseCSV(input)
try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)
try render(rows: rows, comparison: "ft_vs_pt", output: "\(outputDir)/fixed20_signed_entropy_delta_globalfdr_paircompact_ft_vs_pt.png")
try render(rows: rows, comparison: "ft_vs_unseen", output: "\(outputDir)/fixed20_signed_entropy_delta_globalfdr_paircompact_ft_vs_unseen.png")
