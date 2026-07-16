import AppKit
import Foundation

struct Row {
    let comparison: String
    let metric: String
    let layer: Int
    let head: Int
    let pAdj: Double
    let delta: Double
}

let metricOrder: [(csv: String, file: String, label: String)] = [
    ("entropy_delta", "entropy_delta", "Entropy diff."),
    ("l1_mean", "l1_mean", "Mean diff."),
    ("top10_shift_mean", "top10_shift_mean", "Top-10% mean"),
    ("top5_shift_mean", "top5_shift_mean", "Top-5% mean"),
    ("js_div", "js_div", "JSD"),
    ("top1_shift_mean", "top1_shift_mean", "Top-1% mean"),
    ("l2_rms", "l2_rms", "RMSE"),
    ("max_shift", "max_shift", "Max shift"),
]

let tableCounts: [String: [String: Int]] = [
    "entropy_delta": ["ft_vs_pt": 63, "ft_vs_unseen": 69],
    "l1_mean": ["ft_vs_pt": 25, "ft_vs_unseen": 17],
    "top10_shift_mean": ["ft_vs_pt": 24, "ft_vs_unseen": 15],
    "top5_shift_mean": ["ft_vs_pt": 19, "ft_vs_unseen": 13],
    "js_div": ["ft_vs_pt": 18, "ft_vs_unseen": 16],
    "top1_shift_mean": ["ft_vs_pt": 8, "ft_vs_unseen": 6],
    "l2_rms": ["ft_vs_pt": 5, "ft_vs_unseen": 2],
    "max_shift": ["ft_vs_pt": 1, "ft_vs_unseen": 2],
]

func splitCSVLine(_ line: String) -> [String] {
    var result: [String] = []
    var current = ""
    var inQuotes = false
    for ch in line {
        if ch == "\"" {
            inQuotes.toggle()
        } else if ch == "," && !inQuotes {
            result.append(current)
            current = ""
        } else {
            current.append(ch)
        }
    }
    result.append(current)
    return result
}

func parseCSV(_ path: String) throws -> [Row] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    let lines = text.split(whereSeparator: \.isNewline).map(String.init)
    guard let header = lines.first else { return [] }
    let names = splitCSVLine(header)
    func idx(_ name: String) -> Int {
        guard let i = names.firstIndex(of: name) else { fatalError("Missing column: \(name)") }
        return i
    }
    let comparisonI = idx("comparison")
    let metricI = idx("metric")
    let layerI = idx("layer")
    let headI = idx("head")
    let pAdjI = idx("p_adj_bh_global")
    let deltaI = idx("cliffs_delta")

    var rows: [Row] = []
    let keep = Set(metricOrder.map { $0.csv })
    for line in lines.dropFirst() {
        let cols = splitCSVLine(line)
        guard cols.count > deltaI, keep.contains(cols[metricI]) else { continue }
        rows.append(Row(
            comparison: cols[comparisonI],
            metric: cols[metricI],
            layer: Int(cols[layerI]) ?? 0,
            head: Int(cols[headI]) ?? 0,
            pAdj: Double(cols[pAdjI]) ?? 1.0,
            delta: Double(cols[deltaI]) ?? 0.0
        ))
    }
    return rows
}

func colorForDelta(_ value: Double) -> NSColor {
    let v = max(-1.0, min(1.0, value))
    let blue = (49.0, 82.0, 190.0)
    let white = (245.0, 245.0, 245.0)
    let red = (190.0, 20.0, 60.0)
    let rgb: (Double, Double, Double)
    if v < 0 {
        let t = v + 1.0
        rgb = (
            blue.0 + (white.0 - blue.0) * t,
            blue.1 + (white.1 - blue.1) * t,
            blue.2 + (white.2 - blue.2) * t
        )
    } else {
        let t = v
        rgb = (
            white.0 + (red.0 - white.0) * t,
            white.1 + (red.1 - white.1) * t,
            white.2 + (red.2 - white.2) * t
        )
    }
    return NSColor(calibratedRed: rgb.0 / 255.0, green: rgb.1 / 255.0, blue: rgb.2 / 255.0, alpha: 1.0)
}

func drawText(_ text: String, x: CGFloat, y: CGFloat, size: CGFloat, align: NSTextAlignment = .center, bold: Bool = false) {
    let font = bold ? NSFont.boldSystemFont(ofSize: size) : NSFont.systemFont(ofSize: size)
    let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.black]
    let s = NSAttributedString(string: text, attributes: attrs)
    let sizeValue = s.size()
    var drawX = x
    if align == .center { drawX -= sizeValue.width / 2.0 }
    if align == .right { drawX -= sizeValue.width }
    s.draw(at: NSPoint(x: drawX, y: y - sizeValue.height / 2.0))
}

func significantKeys(rows: [Row], metric: String, comparison: String) -> Set<String> {
    let n = tableCounts[metric]?[comparison] ?? 0
    let sortedRows = rows
        .filter { $0.metric == metric && $0.comparison == comparison }
        .sorted {
            if $0.pAdj == $1.pAdj {
                if $0.layer == $1.layer { return $0.head < $1.head }
                return $0.layer < $1.layer
            }
            return $0.pAdj < $1.pAdj
        }
    return Set(sortedRows.prefix(n).map { "\($0.layer)-\($0.head)" })
}

func render(rows: [Row], metric: String, metricLabel: String, comparison: String, output: String) throws {
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
    ) else { fatalError("Failed to allocate bitmap") }

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
    let stars = significantKeys(rows: rows, metric: metric, comparison: comparison)

    var byKey: [String: Row] = [:]
    for r in rows where r.metric == metric && r.comparison == comparison {
        byKey["\(r.layer)-\(r.head)"] = r
    }

    for layer in 0..<nLayers {
        let y = top + CGFloat(layer) * cellH
        for head in 0..<nHeads {
            let key = "\(layer)-\(head)"
            guard let r = byKey[key] else { continue }
            let x = left + CGFloat(head) * cellW
            colorForDelta(r.delta).setFill()
            NSBezierPath(rect: NSRect(x: x, y: y, width: cellW, height: cellH)).fill()
            if stars.contains(key) {
                drawText("*", x: x + cellW / 2.0, y: y + cellH * 0.50, size: 22, bold: true)
            }
        }
    }

    NSColor.black.setStroke()
    let border = NSBezierPath(rect: NSRect(x: left, y: top, width: gridW, height: gridH))
    border.lineWidth = 4
    border.stroke()

    for head in 0..<nHeads {
        drawText("\(head)", x: left + CGFloat(head) * cellW + cellW / 2.0, y: top - 24, size: 22)
    }
    for layer in 0..<nLayers {
        drawText("\(layer)", x: left - 18, y: top + CGFloat(layer) * cellH + cellH / 2.0, size: 18, align: .right)
    }
    drawText("Head", x: left + gridW / 2.0, y: top - 54, size: 28)

    let layerText = NSAttributedString(
        string: "Layer",
        attributes: [.font: NSFont.systemFont(ofSize: 28), .foregroundColor: NSColor.black]
    )
    NSGraphicsContext.current?.cgContext.saveGState()
    NSGraphicsContext.current?.cgContext.translateBy(x: 30, y: top + gridH / 2.0)
    NSGraphicsContext.current?.cgContext.rotate(by: CGFloat.pi / 2.0)
    layerText.draw(at: NSPoint(x: -layerText.size().width / 2.0, y: -layerText.size().height / 2.0))
    NSGraphicsContext.current?.cgContext.restoreGState()

    let cbX = left + gridW + 38
    let cbY = top
    let cbW: CGFloat = 32
    let cbH = gridH
    for i in 0..<160 {
        let v = 1.0 - 2.0 * Double(i) / 159.0
        let y = cbY + CGFloat(159 - i) * cbH / 160.0
        colorForDelta(v).setFill()
        NSBezierPath(rect: NSRect(x: cbX, y: y, width: cbW, height: cbH / 160.0 + 1)).fill()
    }
    let cbBorder = NSBezierPath(rect: NSRect(x: cbX, y: cbY, width: cbW, height: cbH))
    cbBorder.lineWidth = 3
    cbBorder.stroke()
    for val in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let y = cbY + CGFloat((val + 1.0) / 2.0) * cbH
        let tick = NSBezierPath()
        tick.move(to: NSPoint(x: cbX + cbW, y: y))
        tick.line(to: NSPoint(x: cbX + cbW + 10, y: y))
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
    fatalError("Usage: swift render_table_aligned_heatmaps.swift input.csv output_dir")
}
let rows = try parseCSV(args[1])
let outputDir = args[2]
try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

for metric in metricOrder {
    for comparison in ["ft_vs_pt", "ft_vs_unseen"] {
        let output = "\(outputDir)/fixed20_\(metric.file)_table_aligned_\(comparison).png"
        try render(rows: rows, metric: metric.csv, metricLabel: metric.label, comparison: comparison, output: output)
    }
}

var summary = "Feature,FT--PT Sig.,FT--Unseen Sig.\\n"
for metric in metricOrder {
    summary += "\(metric.label),\(tableCounts[metric.csv]?["ft_vs_pt"] ?? 0),\(tableCounts[metric.csv]?["ft_vs_unseen"] ?? 0)\\n"
}
try summary.write(toFile: "\(outputDir)/table_aligned_counts.csv", atomically: true, encoding: .utf8)
