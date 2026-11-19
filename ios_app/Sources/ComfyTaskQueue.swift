import SwiftUI
import Foundation

// MARK: - ComfyUI 任务列表：每个任务的用时与预计完成时间
// 数据来自实例上 ComfyUI 自带的 /api/jobs（创建/开始/结束时间）、/queue（运行与排队任务的工作流）和 /history（已完成任务的工作流）。
// 排队任务的预计用时 = 同配置（主模型、步数、帧数、分辨率、是否二采）已完成任务的中位数用时。

struct ComfyTaskItem: Identifiable, Equatable {
    enum State: Equatable { case running, pending, done, failed }
    let id: String
    var name: String
    var state: State
    var started: Date?
    var ended: Date?
    var estimatedSeconds: Double?
    var eta: Date?
    var stepText: String?
}

private struct ComfyGraphInfo {
    var name: String?
    var classKey: String
    var modelKey: String
    var steps: Int?
}

private func comfyGraphInfo(_ graph: [String: Any]) -> ComfyGraphInfo {
    var name: String?
    var unet = "", steps: Int?, frames = 0, w = 0, h = 0, refine = false, sigmaSteps: Int?
    for (_, v) in graph {
        guard let node = v as? [String: Any], let cls = node["class_type"] as? String,
              let inp = node["inputs"] as? [String: Any] else { continue }
        switch cls {
        case "UNETLoader":
            if unet.isEmpty, let u = inp["unet_name"] as? String { unet = u }
        case "MiniMaxH3Director":
            steps = inp["steps"] as? Int
            frames = inp["total_frames"] as? Int ?? 0
            w = inp["width"] as? Int ?? 0
            h = inp["height"] as? Int ?? 0
            if name == nil, let c = inp["cache_name"] as? String, !c.isEmpty { name = c }
        case "MiniMaxH3DirectorRefine":
            refine = true
        case "ManualSigmas":
            if let s = inp["sigmas"] as? String { sigmaSteps = max(0, s.split(separator: ",").count - 1) }
        case "VHS_VideoCombine", "SaveVideo", "SaveImage":
            if let p = inp["filename_prefix"] as? String, !p.isEmpty {
                name = p.replacingOccurrences(of: "okinawa_", with: "").replacingOccurrences(of: "_hq", with: "")
            }
        default: break
        }
    }
    let realSteps = sigmaSteps ?? steps
    let model = "\(unet)|\(realSteps ?? 0)|\(refine)"
    return ComfyGraphInfo(name: name, classKey: "\(model)|\(frames)|\(w)x\(h)", modelKey: model, steps: realSteps)
}

private func median(_ xs: [Double]) -> Double? {
    guard !xs.isEmpty else { return nil }
    let s = xs.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

private func parseClock(_ s: Substring) -> Double {
    s.split(separator: ":").reduce(0.0) { $0 * 60 + (Double($1) ?? 0) }
}

extension IOSStudioViewModel {
    private func comfyGET(_ path: String) async -> Any? {
        var base = comfyPublicURL.trimmingCharacters(in: .whitespacesAndNewlines)
        while base.hasSuffix("/") { base.removeLast() }
        guard !base.isEmpty, let url = URL(string: base + path),
              let origin = URL(string: base), url.host?.lowercased() == origin.host?.lowercased(),
              origin.user == nil, origin.password == nil, origin.port == nil || origin.port == 443,
              origin.host?.lowercased().hasSuffix(".trycloudflare.com") == true, url.scheme?.lowercased() == "https" else { return nil }
        var req = URLRequest(url: url, timeoutInterval: 8)
        guard let password = RuntimeAccess.password(for: URL(string: normalizedBaseURL)) else { return nil }
        req = RuntimeAccess.authorized(req, password: password)
        guard let (data, resp) = try? await RuntimeAccess.session.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return try? JSONSerialization.jsonObject(with: data)
    }

    /// 采样进度：从运行日志里最新一行 ComfyUI 进度条读出「第几步 / 共几步、剩余时间」
    private func latestSamplingProgress() -> (step: Int, total: Int, remaining: Double)? {
        for line in liveConsoleLines.reversed().prefix(40) where line.contains("[ComfyUI]") {
            guard let r = line.range(of: #"(\d+)/(\d+) \[([\d:]+)<([\d:]+)"#, options: .regularExpression) else { continue }
            let parts = line[r].split(whereSeparator: { " /[<".contains($0) })
            guard parts.count >= 4, let a = Int(parts[0]), let b = Int(parts[1]) else { continue }
            return (a, b, parseClock(parts[3]))
        }
        return nil
    }

    func refreshComfyTasks() async {
        guard comfyRunning, !comfyPublicURL.isEmpty else {
            if !comfyTasks.isEmpty { comfyTasks = [] }
            return
        }
        guard let jobsObj = await comfyGET("/api/jobs?limit=30") as? [String: Any],
              let jobs = jobsObj["jobs"] as? [[String: Any]] else { return }
        let queueObj = await comfyGET("/queue") as? [String: Any] ?? [:]
        let histObj = await comfyGET("/history?max_items=40") as? [String: Any] ?? [:]

        func graphs(_ arr: Any?) -> [String: [String: Any]] {
            var out: [String: [String: Any]] = [:]
            for item in (arr as? [[Any]]) ?? [] where item.count > 2 {
                if let id = item[1] as? String, let g = item[2] as? [String: Any] { out[id] = g }
            }
            return out
        }
        var graphByID = graphs(queueObj["queue_running"]).merging(graphs(queueObj["queue_pending"])) { a, _ in a }
        for (id, v) in histObj {
            if let h = v as? [String: Any], let p = h["prompt"] as? [Any], p.count > 2, let g = p[2] as? [String: Any] { graphByID[id] = g }
        }

        func date(_ v: Any?) -> Date? { (v as? Double).map { Date(timeIntervalSince1970: $0 / 1000) } ?? (v as? Int).map { Date(timeIntervalSince1970: Double($0) / 1000) } }

        // 已完成任务的用时，按配置分组
        var byClass: [String: [Double]] = [:], byModel: [String: [Double]] = [:]
        var lastEnd: Date?
        for j in jobs where (j["status"] as? String) == "completed" {
            guard let s = date(j["execution_start_time"]), let e = date(j["execution_end_time"]) else { continue }
            if lastEnd == nil || e > lastEnd! { lastEnd = e }
            guard let id = j["id"] as? String, let g = graphByID[id] else { continue }
            let info = comfyGraphInfo(g)
            byClass[info.classKey, default: []].append(e.timeIntervalSince(s))
            byModel[info.modelKey, default: []].append(e.timeIntervalSince(s))
        }
        func estimate(_ info: ComfyGraphInfo?) -> Double? {
            guard let info else { return nil }
            return median(byClass[info.classKey] ?? []) ?? median(byModel[info.modelKey] ?? [])
        }

        let now = Date()
        var items: [ComfyTaskItem] = []
        var cursor = now
        // 运行中，然后按排队顺序（priority 小的先跑）
        let active = jobs.filter { ["in_progress", "pending"].contains($0["status"] as? String ?? "") }
            .sorted { ($0["priority"] as? Double ?? 0) < ($1["priority"] as? Double ?? 0) }
        for j in active {
            guard let id = j["id"] as? String else { continue }
            let info = graphByID[id].map(comfyGraphInfo)
            let est = estimate(info)
            let running = (j["status"] as? String) == "in_progress"
            var item = ComfyTaskItem(id: id, name: info?.name ?? String(id.prefix(8)), state: running ? .running : .pending,
                                     started: nil, ended: nil, estimatedSeconds: est, eta: nil, stepText: nil)
            if running {
                let created = date(j["create_time"]) ?? now
                let start = max(created, lastEnd ?? created)
                item.started = start
                let elapsed = now.timeIntervalSince(start)
                var remaining: Double? = est.map { max(15, $0 - elapsed) }
                if let p = latestSamplingProgress(), info?.steps == nil || info?.steps == p.total {
                    item.stepText = L("第 \(p.step)/\(p.total) 步", "step \(p.step)/\(p.total)")
                    remaining = p.remaining + 45   // 采样完还要解码、写文件
                }
                if let r = remaining { cursor = now.addingTimeInterval(r); item.eta = cursor } else { cursor = .distantFuture }
            } else if let est, cursor != .distantFuture {
                cursor = cursor.addingTimeInterval(est)
                item.eta = cursor
            }
            items.append(item)
        }
        // 最近完成 / 失败的 3 个
        let finished = jobs.filter { ["completed", "failed", "error"].contains($0["status"] as? String ?? "") }
            .sorted { (date($0["execution_end_time"]) ?? .distantPast) > (date($1["execution_end_time"]) ?? .distantPast) }
            .prefix(3)
        for j in finished {
            guard let id = j["id"] as? String else { continue }
            let info = graphByID[id].map(comfyGraphInfo)
            items.append(ComfyTaskItem(id: id, name: info?.name ?? String(id.prefix(8)),
                                       state: (j["status"] as? String) == "completed" ? .done : .failed,
                                       started: date(j["execution_start_time"]), ended: date(j["execution_end_time"]),
                                       estimatedSeconds: nil, eta: nil, stepText: nil))
        }
        if items != comfyTasks { comfyTasks = items }
    }
}

struct ComfyTaskListCard: View {
    @ObservedObject var vm: IOSStudioViewModel


    private static let clock: DateFormatter = { let f = DateFormatter(); f.dateFormat = "HH:mm"; return f }()

    private func dur(_ s: Double) -> String {
        let t = Int(s.rounded())
        return t >= 3600 ? String(format: "%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60) : String(format: "%d:%02d", t / 60, t % 60)
    }

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { ctx in
            VStack(alignment: .leading, spacing: 6) {
                Text(L("任务列表", "Tasks"))
                    .font(.system(size: 11.5, weight: .semibold))
                ForEach(vm.comfyTasks) { t in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Image(systemName: icon(t.state))
                            .font(.system(size: 10.5, weight: .semibold))
                            .foregroundStyle(color(t.state))
                        VStack(alignment: .leading, spacing: 1) {
                            Text(t.name)
                                .font(.system(size: 11, weight: .medium))
                                .lineLimit(1).truncationMode(.middle)
                            Text(detail(t, now: ctx.date))
                                .font(.system(size: 10))
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                        }
                        Spacer(minLength: 0)
                    }
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 9)
                .fill(Color(UIColor.secondarySystemGroupedBackground))
                .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(Color.secondary.opacity(0.3), lineWidth: 1))
        )
    }

    private func icon(_ s: ComfyTaskItem.State) -> String {
        switch s { case .running: return "play.circle.fill"; case .pending: return "clock"; case .done: return "checkmark.circle.fill"; case .failed: return "xmark.octagon.fill" }
    }
    private func color(_ s: ComfyTaskItem.State) -> Color {
        switch s { case .running: return .accentColor; case .pending: return .secondary; case .done: return .green; case .failed: return .red }
    }
    private func detail(_ t: ComfyTaskItem, now: Date) -> String {
        let eta = t.eta.map { Self.clock.string(from: $0) } ?? L("未知", "unknown")
        switch t.state {
        case .running:
            let used = t.started.map { dur(now.timeIntervalSince($0)) } ?? "–"
            let step = t.stepText.map { " · " + $0 } ?? ""
            return L("运行中 · 已用 \(used)\(step) · 预计 \(eta) 完成", "Running · \(used) elapsed\(step) · ETA \(eta)")
        case .pending:
            let est = t.estimatedSeconds.map { "~" + dur($0) } ?? L("未知", "unknown")
            return L("排队 · 预计用时 \(est) · 预计 \(eta) 完成", "Queued · est. \(est) · ETA \(eta)")
        case .done:
            let used = (t.started != nil && t.ended != nil) ? dur(t.ended!.timeIntervalSince(t.started!)) : "–"
            let end = t.ended.map { Self.clock.string(from: $0) } ?? "–"
            return L("已完成 · 用时 \(used) · \(end) 完成", "Done · took \(used) · finished \(end)")
        case .failed:
            let end = t.ended.map { Self.clock.string(from: $0) } ?? "–"
            return L("失败 · \(end)", "Failed · \(end)")
        }
    }
}
