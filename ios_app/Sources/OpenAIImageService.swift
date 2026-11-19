import SwiftUI
import UIKit
import Security

// MARK: - GPT-Image：客户端直连 OpenAI 图像接口（不经过 Colab，不需要实例，API Key 只存在本机钥匙串）

let GPT_IMAGE_MODEL_ID = "gpt_image"
let GPT_IMAGE_MAX_REFERENCES = 16
// OpenAI 文档里 2.6 与 2.5 的模型名并存；填好 Key 后会改用账号实际可用的列表（见 refreshOpenAIImageModels）
let GPT_IMAGE_DEFAULT_MODELS = ["gpt-image-2.6-sunburst", "gpt-image-2.6-flare", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare"]
let GPT_IMAGE_QUALITIES: [(id: String, zh: String, en: String)] = [
    ("auto", "自动", "Auto"), ("low", "低", "Low"), ("medium", "标准", "Medium"),
    ("high", "高", "High"), ("xhigh", "超高", "X-High"), ("max", "极致", "Max")
]
let GPT_IMAGE_COMPLEXITIES: [(id: String, zh: String, en: String)] = [("low", "低", "Low"), ("medium", "中", "Medium"), ("high", "高", "High")]
let GPT_IMAGE_ASPECTS: [(id: String, w: Int, h: Int)] = [
    ("1:1", 1, 1), ("1:2", 1, 2), ("2:1", 2, 1), ("1:3", 1, 3), ("3:1", 3, 1),
    ("2:3", 2, 3), ("3:2", 3, 2), ("3:4", 3, 4), ("4:3", 4, 3), ("4:5", 4, 5),
    ("5:4", 5, 4), ("9:16", 9, 16), ("16:9", 16, 9), ("21:9", 21, 9), ("9:21", 9, 21)
]
// 档位 = (像素预算, 单边上限)。4K 档已顶到接口上限：单边 3840、总像素 8,294,400（超过 2560×1440 官方标为实验性）。
// 2026-11-11 实测这两条都是硬限制：3072×4096 报 "longest edge must be <= 3840"，2880×3840 报 "exceeds the current pixel budget"
let GPT_IMAGE_TIERS: [(id: String, pixels: Int, maxEdge: Int)] = [
    ("1K", 1_572_864, 3840), ("2K", 3_686_400, 3840), ("4K", 8_294_400, 3840)
]

/// 接口要求宽、高都是 16 的倍数：从预算允许的最大宽度往下找第一个能保持精确比例的尺寸
func gptImageSize(aspect: String, tier: String) -> (width: Int, height: Int) {
    let a = GPT_IMAGE_ASPECTS.first { $0.id == aspect } ?? GPT_IMAGE_ASPECTS[0]
    let t = GPT_IMAGE_TIERS.first { $0.id == tier } ?? GPT_IMAGE_TIERS[0]
    let widthForMaxEdge = a.w >= a.h ? t.maxEdge : t.maxEdge * a.w / a.h
    var w = min(Int((Double(t.pixels) * Double(a.w) / Double(a.h)).squareRoot()), widthForMaxEdge) / 16 * 16
    var rounded: (Int, Int)? = nil
    for _ in 0..<40 {
        let h = w * a.h / a.w
        if w * h <= t.pixels && h <= t.maxEdge {
            if (w * a.h) % a.w == 0 && h % 16 == 0 { return (w, h) }
            if rounded == nil { rounded = (w, h / 16 * 16) }
        }
        w -= 16
    }
    return rounded ?? (1024, 1024)
}

let GPT_IMAGE_AUTO_ASPECT = "auto"
let GPT_IMAGE_OUTPUT_USD_PER_MILLION_TOKENS = 30.0

/// 一张图的输出 token 估算，公式取自 OpenAI 图像生成指南里的成本计算器（Calculating costs）。
/// 只覆盖 gpt-image-2.5 / 2.6；画质为 auto 时接口自己定档，无法预估，返回 nil
func gptImageEstimatedOutputTokens(width: Int, height: Int, quality: String, model: String) -> Int? {
    let tiles: [String: Double] = ["low": 16, "medium": 24, "high": 48, "xhigh": 64, "max": 96]
    guard let longSide = tiles[quality], model.contains("2.6") || model.contains("2.5") else { return nil }
    let ratio = longSide / (Double(max(width, height)) / Double(min(width, height)))
    let floorValue = ratio.rounded(.down)
    let shortSide = ratio - floorValue == 0.5 ? floorValue + floorValue.truncatingRemainder(dividingBy: 2) : (ratio + 0.5).rounded(.down)
    let tokens = (longSide * shortSide * (2_000_000 + Double(width * height)) / 4_000_000).rounded(.up)
    // 同一档位下 2.6 的低、中两档比 2.5 多用 token
    let factors26: [String: Double] = ["low": 4.2, "medium": 3.3]
    let factor = model.contains("2.6") ? (factors26[quality] ?? 1.0) : 1.0
    return Int((tokens * factor + 0.5).rounded(.down))
}

/// 发给接口的 size 字段。"auto" 表示让模型按提示词自己定画幅，此时分辨率档不生效
func gptImageSizeField(aspect: String, tier: String) -> String {
    if aspect == GPT_IMAGE_AUTO_ASPECT { return "auto" }
    let size = gptImageSize(aspect: aspect, tier: tier)
    return "\(size.width)x\(size.height)"
}

/// 每条提示词的第一行如果只是在说画幅（比例、分辨率、横竖），就按它来，这一行不发给模型。
/// 写法不限：`16:9 2K图`、`竖版 4K`、`[3:4 4K]`、`3比4 的 4k 图片`、`自动` 都认，括号可有可无、全角半角都行。
/// 这一行里只要还有别的实质内容（比如一句完整的描述），就整段当作提示词正文，不做猜测。
func parseGPTPromptTag(_ prompt: String) -> (aspect: String?, tier: String?, text: String) {
    let lines = prompt.components(separatedBy: .newlines)
    guard lines.count > 1 else { return (nil, nil, prompt) }
    var rest = lines[0].applyingTransform(.fullwidthToHalfwidth, reverse: false) ?? lines[0]
    var aspect: String? = nil
    var tier: String? = nil

    func take(_ pattern: String, _ handler: ([String]) -> Bool) {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
              let match = regex.firstMatch(in: rest, range: NSRange(rest.startIndex..., in: rest)) else { return }
        let groups = (0..<match.numberOfRanges).map { Range(match.range(at: $0), in: rest).map { String(rest[$0]) } ?? "" }
        if handler(groups), let whole = Range(match.range, in: rest) { rest.removeSubrange(whole) }
    }
    take("(\\d{1,2})\\s*(?::|比|/|x|×)\\s*(\\d{1,2})") { g in
        let id = "\(g[1]):\(g[2])"
        guard GPT_IMAGE_ASPECTS.contains(where: { $0.id == id }) else { return false }
        aspect = id
        return true
    }
    take("([124])\\s*k") { g in tier = "\(g[1])K"; return true }
    let words: [(String, String)] = [
        ("横版|横图|横屏|横向|landscape", "16:9"), ("竖版|竖图|竖屏|竖向|portrait", "9:16"),
        ("正方形|方图|方形|square", "1:1"), ("自动|auto", GPT_IMAGE_AUTO_ASPECT)
    ]
    for (pattern, id) in words where aspect == nil {
        take(pattern) { _ in aspect = id; return true }
    }
    // 去掉括号、标点和"图、的、比例"这类虚词后还剩东西，说明这一行是正文而不是画幅说明
    let filler = "图片|图像|宽高比|分辨率|比例|尺寸|画幅|大小|生成|一张|图|的|张|出|用|要|按|来|幅|[\\[\\]【】()（）,，、。.;；:：\\s]"
    let leftover = rest.replacingOccurrences(of: filler, with: "", options: [.regularExpression, .caseInsensitive])
    let text = lines.dropFirst().joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    guard leftover.isEmpty, aspect != nil || tier != nil, !text.isEmpty else { return (nil, nil, prompt) }
    return (aspect, tier, text)
}

enum OpenAIKeychain {
    private static let query: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: "app.easycolab.studio.ios.openai",
        kSecAttrAccount as String: "api_key"
    ]

    static func read() -> String? {
        var q = query
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &item) == errSecSuccess, let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func save(_ key: String) -> Bool {
        SecItemDelete(query as CFDictionary)
        var q = query
        q[kSecValueData as String] = Data(key.utf8)
        return SecItemAdd(q as CFDictionary, nil) == errSecSuccess
    }

    static func delete() {
        SecItemDelete(query as CFDictionary)
    }
}

extension IOSStudioViewModel {
    var isCloudAPIModel: Bool { selectedModelId == GPT_IMAGE_MODEL_ID }

    func saveOpenAIKey() {
        let key = openAIKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { return }
        guard OpenAIKeychain.save(key) else {
            statusMessage = L("API Key 写入钥匙串失败，请重试。", "Failed to store the API key in Keychain. Please try again.")
            return
        }
        openAIKeyDraft = ""
        openAIKeyConfigured = true
        statusMessage = L("OpenAI API Key 已保存到本机钥匙串，正在查询账号可用的图像模型...", "OpenAI API key saved to Keychain. Checking which image models your account can use...")
        Task { await refreshOpenAIImageModels() }
    }

    func deleteOpenAIKey() {
        OpenAIKeychain.delete()
        openAIKeyConfigured = false
        statusMessage = L("已从钥匙串移除 OpenAI API Key。", "Removed the OpenAI API key from Keychain.")
    }

    /// 文档里的模型名不一定都对账号开放，所以以 /v1/models 实际返回的 gpt-image-* 为准
    func refreshOpenAIImageModels() async {
        guard let key = OpenAIKeychain.read(), let url = URL(string: "https://api.openai.com/v1/models") else { return }
        var req = URLRequest(url: url)
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 30
        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard code == 200, let list = json?["data"] as? [[String: Any]] else {
                statusMessage = L(
                    "查询 OpenAI 模型列表失败：\(openAIErrorMessage(json, code: code))",
                    "Failed to list OpenAI models: \(openAIErrorMessage(json, code: code))"
                )
                return
            }
            let ids = list.compactMap { $0["id"] as? String }.filter { $0.hasPrefix("gpt-image") }.sorted(by: >)
            guard !ids.isEmpty else {
                statusMessage = L(
                    "API Key 有效，但账号下没有 gpt-image 模型（可能需要先在 OpenAI 后台完成组织验证）。",
                    "The key works, but this account has no gpt-image models (organization verification may be required)."
                )
                return
            }
            openAIImageModels = ids
            if !ids.contains(openAIImageModel) { openAIImageModel = ids[0] }
            statusMessage = L("账号可用的图像模型共 \(ids.count) 个，当前选择 \(openAIImageModel)。", "\(ids.count) image models available; using \(openAIImageModel).")
        } catch {
            statusMessage = L("查询 OpenAI 模型列表失败：\(error.localizedDescription)", "Failed to list OpenAI models: \(error.localizedDescription)")
        }
    }

    func cancelOpenAIImageGeneration() {
        openAIRequestTask?.cancel()
        openAIRequestTask = nil
        activeGenerationToken = UUID()
        stopLiveGenerationTimer()
        isGenerating = false
        progressValue = 0.0
        currentJobElapsed = 0.0
        statusMessage = L(
            "已停止等待。已经发给 OpenAI 的那一张可能仍会计费。",
            "Stopped waiting. The request already sent to OpenAI may still be billed."
        )
    }

    /// 单独一行的 ---（三个以上减号）把输入切成多个提示词，每个提示词各自出图
    func splitGPTPrompts(_ text: String) -> [String] {
        var prompts: [String] = []
        var current: [String] = []
        for line in text.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.count >= 3 && trimmed.allSatisfy({ $0 == "-" }) {
                prompts.append(current.joined(separator: "\n"))
                current = []
            } else {
                current.append(line)
            }
        }
        prompts.append(current.joined(separator: "\n"))
        return prompts.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }

    func startOpenAIImageGeneration(prompt: String) async {
        guard let key = OpenAIKeychain.read(), !key.isEmpty else {
            openAIKeyConfigured = false
            statusMessage = L("请先在左侧填写并保存 OpenAI API Key。", "Enter and save your OpenAI API key on the left first.")
            return
        }
        let prompts = splitGPTPrompts(prompt)
        let isEdit = taskModeId == "i2i_edit"
        var base: [String: Any] = [
            "model": openAIImageModel,
            "quality": openAIImageQuality,
            "output_format": "png"
        ]
        // 不传时模型默认就是 high；只在调低时才带上这个字段，避免不支持它的旧模型报错
        if openAIImageComplexity != "high" { base["complexity"] = openAIImageComplexity }
        var refCount = 0
        if isEdit {
            // 参考图按"主参考图、参考图 2、追加参考图"的顺序发送
            let refs = ([firstFrameImage, lastFrameImage].compactMap { $0 } + extraReferenceImages).prefix(GPT_IMAGE_MAX_REFERENCES)
            let encoded = refs.compactMap { encodeImageToBase64($0) }.map { ["image_url": "data:image/jpeg;base64,\($0)"] }
            base["images"] = encoded
            refCount = encoded.count
        }
        guard !prompts.isEmpty, let url = URL(string: "https://api.openai.com/v1/images/\(isEdit ? "edits" : "generations")") else { return }

        let myToken = UUID()
        activeGenerationToken = myToken
        isGenerating = true
        progressValue = 0.0
        currentJobElapsed = 0.0
        startLiveGenerationTimer()
        let perPrompt = batchCount
        let total = prompts.count * perPrompt
        let stamp = DateFormatter()
        stamp.dateFormat = "yyyyMMdd_HHmmss"
        var done = 0
        var failed = 0
        var lastError = ""

        requests: for (promptIndex, onePrompt) in prompts.enumerated() {
            for _ in 0..<perPrompt {
                let ordinal = done + failed + 1
                let prefix = total > 1 ? "[\(ordinal)/\(total)] " : ""
                progressValue = (Double(ordinal) - 0.5) / Double(total)
                statusMessage = L(
                    "\(prefix)已提交给 OpenAI (\(openAIImageModel))，正在等待出图...",
                    "\(prefix)Sent to OpenAI (\(openAIImageModel)); waiting for the image..."
                )
                // 这一条提示词自带画幅标记时用它的，否则用左侧面板选的
                let tag = parseGPTPromptTag(onePrompt)
                let sizeField = gptImageSizeField(aspect: tag.aspect ?? gptImageAspect, tier: tag.tier ?? gptImageTier)
                var body = base
                body["prompt"] = tag.text
                body["size"] = sizeField
                var req = URLRequest(url: url)
                req.httpMethod = "POST"
                req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                req.timeoutInterval = 900
                req.httpBody = try? JSONSerialization.data(withJSONObject: body)
                let startedAt = Date()
                logOpenAI("\(prefix)请求 POST \(url.path) · model=\(openAIImageModel) · size=\(sizeField) · quality=\(openAIImageQuality) · complexity=\(openAIImageComplexity)"
                          + (prompts.count > 1 ? " · 提示词 \(promptIndex + 1)/\(prompts.count)" : "")
                          + (isEdit ? " · 参考图 \(refCount) 张" : ""))
                do {
                    // 有些代理会掐掉空闲约 2 分钟的连接，而出图常要等更久。等待期间每 45 秒在同一条
                    // HTTP/2 连接上发一个很小的请求，让它一直有流量；实测这样连接能撑过掐断点。
                    let keepalive = Task {
                        guard let pingURL = URL(string: "https://api.openai.com/v1/models") else { return }
                        var ping = URLRequest(url: pingURL)
                        ping.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
                        ping.timeoutInterval = 20
                        while !Task.isCancelled {
                            try? await Task.sleep(nanoseconds: 45_000_000_000)
                            if Task.isCancelled { break }
                            _ = try? await URLSession.shared.data(for: ping)
                        }
                    }
                    defer { keepalive.cancel() }
                    let task = Task { try await URLSession.shared.data(for: req) }
                    openAIRequestTask = task
                    let (data, resp) = try await task.value
                    guard activeGenerationToken == myToken else { return }
                    let http = resp as? HTTPURLResponse
                    let code = http?.statusCode ?? 0
                    let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                    // 只记状态、耗时、请求号和用量，不记 Key 和图片数据；请求号可用于向 OpenAI 查询或核对账单
                    var meta = "HTTP \(code) · 耗时 \(String(format: "%.1f", Date().timeIntervalSince(startedAt)))s"
                    if let rid = http?.value(forHTTPHeaderField: "x-request-id") { meta += " · request-id=\(rid)" }
                    guard code == 200,
                          let b64 = ((json?["data"] as? [[String: Any]])?.first?["b64_json"]) as? String,
                          let png = Data(base64Encoded: b64) else {
                        logOpenAI("\(prefix)失败 \(meta) · \(openAIErrorMessage(json, code: code))")
                        throw NSError(domain: "OpenAI", code: code, userInfo: [NSLocalizedDescriptionKey: openAIErrorMessage(json, code: code)])
                    }
                    if let actualSize = json?["size"] as? String { meta += " · 实际尺寸 \(actualSize)" }
                    if let quality = json?["quality"] as? String { meta += " · 实际质量 \(quality)" }
                    if let usage = json?["usage"] as? [String: Any] {
                        meta += " · tokens 输入 \(usage["input_tokens"] ?? "?") / 输出 \(usage["output_tokens"] ?? "?") / 合计 \(usage["total_tokens"] ?? "?")"
                        // 只按输出 token 折算；输入的文字和参考图另按各自单价计费，金额以 OpenAI 账单为准
                        if let output = usage["output_tokens"] as? Int {
                            meta += " · 输出约 $\(String(format: "%.3f", Double(output) * GPT_IMAGE_OUTPUT_USD_PER_MILLION_TOKENS / 1_000_000))"
                        }
                    }
                    let suffix = total > 1 ? "_\(ordinal)" : ""
                    let destURL = localSaveDir.appendingPathComponent("gpt_image_\(isEdit ? "edit" : "t2i")_\(stamp.string(from: Date()))\(suffix).png")
                    try png.write(to: destURL)
                    mediaElapsedMap[destURL.lastPathComponent] = Date().timeIntervalSince(startedAt)
                    refreshLocalMedia()
                    selectMedia(destURL)
                    logOpenAI("\(prefix)返回 \(meta) · 已保存 \(destURL.lastPathComponent)")
                    done += 1
                } catch {
                    guard activeGenerationToken == myToken else { return }
                    // HTTP 层面的失败上面已经记过；这里补记断网、超时这类没有拿到响应的情况
                    if (error as NSError).domain != "OpenAI" { logOpenAI("\(prefix)失败 · \(error.localizedDescription)") }
                    failed += 1
                    lastError = error.localizedDescription
                    // 一张失败不影响队列里其余的；但 Key 无效时后面只会全部失败，直接停
                    if (error as NSError).code == 401 { break requests }
                }
            }
        }

        openAIRequestTask = nil
        stopLiveGenerationTimer()
        isGenerating = false
        if done == 0 {
            progressValue = 0.0
            statusMessage = L("GPT-Image 生成失败：\(lastError)", "GPT-Image failed: \(lastError)")
            return
        }
        progressValue = 1.0
        lastCompletedElapsed = currentJobElapsed
        statusMessage = failed > 0
            ? L("GPT-Image 完成 \(done) 张，失败 \(failed) 张（最后一个错误：\(lastError)）。", "GPT-Image finished \(done), failed \(failed) (last error: \(lastError)).")
            : (total > 1
                ? L("GPT-Image 已生成 \(total) 张，保存在本地输出目录。", "GPT-Image generated \(total) images; saved to the local output folder.")
                : L("GPT-Image 生成完成，已保存到本地输出目录。", "GPT-Image done; saved to the local output folder."))
    }

    /// 写到右侧"运行日志"面板，同时追加到本地输出目录的 gpt_image_log.txt，方便事后核对
    func logOpenAI(_ text: String) {
        let stamp = DateFormatter()
        stamp.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let now = stamp.string(from: Date())
        let line = "[\(now.suffix(8))] [GPT-Image] \(text)"
        openAILogLines = Array((openAILogLines + [line]).suffix(200))
        liveConsoleLines.append(line)
        let fileURL = localSaveDir.appendingPathComponent("gpt_image_log.txt")
        let data = Data("[\(now)] \(text)\n".utf8)
        if let handle = try? FileHandle(forWritingTo: fileURL) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: fileURL)
        }
    }

    private func openAIErrorMessage(_ json: [String: Any]?, code: Int) -> String {
        ((json?["error"] as? [String: Any])?["message"] as? String) ?? "HTTP \(code)"
    }
}
