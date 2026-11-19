import SwiftUI
import UIKit
import Foundation

// MARK: - Bilingual Localization Manager (Default: Simplified Chinese "zh")

enum AppLanguage: String, CaseIterable {
    case zh = "zh"
    case en = "en"
}

extension Notification.Name {
    static let easyColabLanguageDidChange = Notification.Name("EasyColabIOS.LanguageDidChange")
}

final class LanguageManager: ObservableObject {
    static let shared = LanguageManager()

    @Published var lang: AppLanguage {
        didSet {
            UserDefaults.standard.set(lang.rawValue, forKey: "appLanguage")
            if oldValue != lang {
                NotificationCenter.default.post(name: .easyColabLanguageDidChange, object: nil)
            }
        }
    }

    var isEN: Bool { lang == .en }

    private init() {
        if UserDefaults.standard.string(forKey: "appLanguage") == "en" {
            self.lang = .en
        } else {
            self.lang = .zh
        }
    }

    func toggle() {
        lang = (lang == .zh) ? .en : .zh
    }
}

@inline(__always)
func L(_ zh: String, _ en: String) -> String {
    LanguageManager.shared.lang == .en ? en : zh
}

// MARK: - Global iOS Keyboard Dismissal Helper, Accessory Bar & Tap-Outside Installer

@MainActor
func dismissIOSKeyboard() {
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
}

@MainActor
final class IOSKeyboardObserver: ObservableObject {
    static let shared = IOSKeyboardObserver()

    @Published private(set) var isVisible: Bool = false

    private init() {
        let nc = NotificationCenter.default
        nc.addObserver(forName: UIResponder.keyboardWillShowNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.isVisible = true }
        }
        nc.addObserver(forName: UIResponder.keyboardWillHideNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.isVisible = false }
        }
        nc.addObserver(forName: UITextField.textDidBeginEditingNotification, object: nil, queue: .main) { [weak self] note in
            guard let tf = note.object as? UITextField else { return }
            Task { @MainActor in self?.ensureAccessoryToolbar(on: tf) }
        }
        nc.addObserver(forName: UITextView.textDidBeginEditingNotification, object: nil, queue: .main) { [weak self] note in
            guard let tv = note.object as? UITextView, tv.isEditable else { return }
            Task { @MainActor in self?.ensureAccessoryToolbar(on: tv) }
        }
    }

    private func makeAccessoryToolbar(tag: Int) -> UIToolbar {
        let bar = UIToolbar(frame: CGRect(x: 0, y: 0, width: UIScreen.main.bounds.width, height: 42))
        bar.tag = tag
        bar.sizeToFit()

        let flex = UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)

        var btnConfig = UIButton.Configuration.gray()
        btnConfig.cornerStyle = .capsule
        btnConfig.baseForegroundColor = .label
        btnConfig.image = UIImage(
            systemName: "keyboard.chevron.compact.down",
            withConfiguration: UIImage.SymbolConfiguration(pointSize: 12, weight: .bold)
        )
        btnConfig.imagePadding = 5
        btnConfig.contentInsets = NSDirectionalEdgeInsets(top: 5, leading: 12, bottom: 5, trailing: 12)
        var attrTitle = AttributedString(L("收起键盘", "Done"))
        attrTitle.font = .system(size: 13, weight: .bold)
        btnConfig.attributedTitle = attrTitle

        let btn = UIButton(configuration: btnConfig, primaryAction: UIAction { _ in
            Task { @MainActor in dismissIOSKeyboard() }
        })
        let doneItem = UIBarButtonItem(customView: btn)
        bar.items = [flex, doneItem]
        return bar
    }

    private func ensureAccessoryToolbar(on textField: UITextField) {
        let langTag = LanguageManager.shared.isEN ? 202 : 101
        if textField.inputAccessoryView?.tag == langTag { return }
        textField.inputAccessoryView = makeAccessoryToolbar(tag: langTag)
        textField.reloadInputViews()
    }

    private func ensureAccessoryToolbar(on textView: UITextView) {
        let langTag = LanguageManager.shared.isEN ? 202 : 101
        if textView.inputAccessoryView?.tag == langTag { return }
        textView.inputAccessoryView = makeAccessoryToolbar(tag: langTag)
        textView.reloadInputViews()
    }
}

struct IOSKeyboardDismissTapInstaller: UIViewRepresentable {
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> UIView {
        let view = InstallerHostView()
        view.backgroundColor = .clear
        view.isUserInteractionEnabled = false
        view.onMoveToWindowCallback = { [weak coordinator = context.coordinator] window in
            coordinator?.installIfNeeded(on: window)
        }
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        if let window = uiView.window {
            context.coordinator.installIfNeeded(on: window)
        }
    }

    final class InstallerHostView: UIView {
        var onMoveToWindowCallback: ((UIWindow?) -> Void)?
        override func didMoveToWindow() {
            super.didMoveToWindow()
            onMoveToWindowCallback?(window)
        }
    }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        private static let gestureName = "EasyColabIOSKeyboardDismissTap"

        func installIfNeeded(on window: UIWindow?) {
            guard let window else { return }
            if window.gestureRecognizers?.contains(where: { $0.name == Self.gestureName }) == true {
                return
            }
            let tap = UITapGestureRecognizer(target: self, action: #selector(handleBackgroundTap(_:)))
            tap.name = Self.gestureName
            tap.cancelsTouchesInView = false
            tap.delaysTouchesBegan = false
            tap.delaysTouchesEnded = false
            tap.delegate = self
            window.addGestureRecognizer(tap)
        }

        @objc private func handleBackgroundTap(_ recognizer: UITapGestureRecognizer) {
            guard recognizer.state == .ended else { return }
            Task { @MainActor in
                dismissIOSKeyboard()
            }
        }

        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldReceive touch: UITouch) -> Bool {
            guard IOSKeyboardObserver.shared.isVisible else { return false }
            var current: UIView? = touch.view
            while let view = current {
                if view is UITextField || view is UITextView || view is UIControl {
                    return false
                }
                let typeName = String(describing: type(of: view))
                if typeName.contains("TextField")
                    || typeName.contains("TextView")
                    || typeName.contains("TextEditor")
                    || typeName.contains("Keyboard")
                    || typeName.contains("ContextMenu")
                    || typeName.contains("Button") {
                    return false
                }
                current = view.superview
            }
            return true
        }

        func gestureRecognizer(
            _ gestureRecognizer: UIGestureRecognizer,
            shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer
        ) -> Bool {
            true
        }
    }
}

func localizeModelFileDesc(_ raw: String) -> String {
    guard LanguageManager.shared.isEN, !raw.isEmpty else { return raw }
    var s = raw
    s = s.replacingOccurrences(of: " [RAM内存盘]", with: " [RAM Disk]")
    s = s.replacingOccurrences(of: "DiT 音视频主干 BF16", with: "DiT Audio-Video Backbone BF16")
    s = s.replacingOccurrences(of: "多模态编码器 BF16", with: "Multimodal Encoder BF16")
    s = s.replacingOccurrences(of: "3D 视频 VAE BF16", with: "3D Video VAE BF16")
    s = s.replacingOccurrences(of: "32kHz 立体声 VAE BF16", with: "32kHz Stereo Audio VAE BF16")
    s = s.replacingOccurrences(of: "权重分片", with: "Weight Shard")
    s = s.replacingOccurrences(of: "Unsloth UD-Q4_K_M 主模型", with: "Unsloth UD-Q4_K_M Main Model")
    s = s.replacingOccurrences(of: "多模态视觉投影器", with: "Multimodal Vision Projector")
    s = s.replacingOccurrences(of: "Unsloth MTP 多词元投机解码头", with: "Unsloth MTP Speculative Decoding Head")
    s = s.replacingOccurrences(of: "编码器", with: "Encoder")
    s = s.replacingOccurrences(of: "图像 VAE", with: "Image VAE")
    return s
}

func localizeServerText(_ text: String) -> String {
    guard !text.isEmpty else { return text }
    let adaptedZh = text
        .replacingOccurrences(of: "Mac 客户端", with: "App 客户端")
        .replacingOccurrences(of: "左侧", with: "算力页/上方")
    guard LanguageManager.shared.isEN else { return adaptedZh }

    let exactMap: [String: String] = [
        "未加载": "Not Loaded",
        "未连接 Colab 服务端": "Colab Server Disconnected",
        "尚未启动云端 ComfyUI 工作台": "Cloud ComfyUI Studio Not Started",
        "Google Drive 自动软链持久化": "Google Drive Auto-Symlink Persistence",
        "Google Drive 实时软链挂载": "Google Drive Live Symlink Mount",
        "Google Drive 云盘自动持久化": "Google Drive Auto-Persistence",
        "实例本地高速存储": "Instance Local NVMe Storage",
        "已彻底释放全部云端实例，计费速率归零。": "All cloud Colab instances terminated. Billing rate is now 0 CU/h.",
        "已手动停止当前推理任务": "Inference cancelled by user",
        "正在上传最新多模型开源服务端代码 (app.py)...": "Uploading latest multi-model server engine (app.py)...",
        "正在云端配置 Google Drive 持久化、多模型引擎并启动隧道 (约需 40~90 秒)...": "Configuring Google Drive persistence, multi-model engine & public tunnel (~40–90s)...",
        "正在克隆 DiffSynth-Studio...": "Cloning DiffSynth-Studio..."
    ]
    if let exact = exactMap[text] { return exact }

    var s = localizeModelFileDesc(text)
    let replacements: [(String, String)] = [
        ("已无缝接管在线 ComfyUI 画布", "Connected to active ComfyUI canvas"),
        ("ComfyUI 进程已退出", "ComfyUI process exited"),
        ("，可点击重新启动", "; tap to restart"),
        ("正在 8 线程极速下载 ", "8-thread fast downloading "),
        ("正在 8 线程并发下载 ", "8-thread downloading "),
        ("正在下载 ", "Downloading "),
        (" 已就绪 (", " ready ("),
        (")，跳过下载", "), skipping"),
        ("正在评估显存余量并装载 ", "Checking VRAM & loading "),
        (" 至 GPU...", " into GPU..."),
        (" 正在编码多模态上下文并启动推理...", " encoding multimodal context & starting inference..."),
        (" 推理完成！耗时 ", " inference complete! Elapsed "),
        (" 正在使用多模态编码器解析提示词", " encoding prompts with multimodal encoder"),
        ("DiT 采样已完成，正在执行", "DiT sampling finished, running "),
        ("3D 视频 VAE 解码", "3D Video VAE decoding"),
        ("视频帧解码完成，正在执行 32kHz 立体声音轨解码", "Video frames decoded, decoding 32kHz stereo audio"),
        (" 正在执行去噪采样第 ", " denoising step "),
        (" 步 (", " ("),
        ("VAE 解码完成，正在合成 32kHz 立体声音轨并封装 MP4", "VAE decoded, muxing 32kHz stereo audio into MP4"),
        ("正在写入无损 PNG 图像文件", "Writing lossless PNG image"),
        ("生成完成！耗时 ", "Generation complete! Elapsed "),
        ("生成出错: ", "Generation error: "),
        ("尚未加载至 Colab GPU 显存，请先在 Mac 客户端点击「加载到显存」。", "is not loaded in Colab GPU VRAM. Tap 'Load to VRAM' first."),
        ("尚未加载到显存", "is not loaded in VRAM"),
        ("当前已常驻：", "currently resident: "),
        ("已通过 Google OAuth 自动挂载 Google Drive", "Mounted Google Drive automatically via Google OAuth")
    ]
    for (zhSub, enSub) in replacements where s.contains(zhSub) {
        s = s.replacingOccurrences(of: zhSub, with: enSub)
    }
    return s
}

func stripEmoji(_ text: String) -> String {
    var scalars = String.UnicodeScalarView()
    for s in text.unicodeScalars {
        let p = s.properties
        let v = s.value
        let isEmoji = p.isEmojiPresentation
            || (p.isEmoji && v > 0x238C)
            || v == 0xFE0F || v == 0x200D || v == 0x20E3
            || (0x1F1E6...0x1F1FF).contains(v)
        if !isEmoji { scalars.append(s) }
    }
    var result = String(scalars)
    while result.contains("  ") { result = result.replacingOccurrences(of: "  ", with: " ") }
    return result.trimmingCharacters(in: .whitespaces)
}

func formatDuration(_ sec: Double, compact: Bool = false) -> String {
    let s = max(0.0, sec)
    if s < 60.0 {
        if compact { return String(format: "%.1fs", s) }
        return LanguageManager.shared.isEN
            ? String(format: "%.1f s", s)
            : String(format: "%.1f 秒", s)
    }
    let mins = Int(s) / 60
    let rem = s - Double(mins * 60)
    if compact { return String(format: "%dm%.0fs", mins, rem) }
    return LanguageManager.shared.isEN
        ? String(format: "%dm %.1fs", mins, rem)
        : String(format: "%d分%.1f秒", mins, rem)
}

extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        min(max(self, limits.lowerBound), limits.upperBound)
    }
}

// MARK: - Multimodal Models & Presets Catalog

enum ModelModality: String, Codable {
    case videoAudio = "video_audio"
    case video = "video"
    case image = "image"
    case llmVLM = "llm_vlm"
    case comfyUI = "comfyui"
}

struct TaskModeItem: Identifiable, Hashable {
    let id: String
    let titleZh: String
    let titleEn: String
    let subtitleZh: String
    let subtitleEn: String
    let icon: String

    var title: String { L(titleZh, titleEn) }
    var subtitle: String { L(subtitleZh, subtitleEn) }
}

struct ModelFileItem: Codable, Identifiable, Hashable {
    var id: String { name }
    var name: String
    var desc: String?
    var exists: Bool
    var size_gb: Double

    var localizedDesc: String? {
        guard let raw = desc, !raw.isEmpty else { return nil }
        return localizeModelFileDesc(raw)
    }
}

struct CustomHFModelDTO: Codable, Identifiable, Hashable {
    var id: String
    var name: String
    var repo_id: String
    var modality: String
    var category: String?
    var approx_size_gb: Double?
    var include_pattern: String?
    var pipeline_arch: String?
    var files_count: Int?
    var is_custom: Bool?
}

struct OpenModelSpec: Identifiable, Hashable {
    let id: String
    let nameZh: String
    let nameEn: String
    var shortTitleZh: String = ""
    var shortTitleEn: String = ""
    let repoId: String
    let badgeZh: String
    let badgeEn: String
    let summaryZh: String
    let summaryEn: String
    let approxSizeGB: Double
    let icon: String
    let modality: ModelModality
    let defaultSteps: Double
    let modes: [TaskModeItem]
    let defaultFiles: [ModelFileItem]
    var includePattern: String = ""
    var isCustom: Bool = false

    var name: String { L(nameZh, nameEn) }
    var shortTitle: String { L(shortTitleZh.isEmpty ? nameZh : shortTitleZh, shortTitleEn.isEmpty ? nameEn : shortTitleEn) }
    var badge: String { L(badgeZh, badgeEn) }
    var summary: String { L(summaryZh, summaryEn) }

    static func == (lhs: OpenModelSpec, rhs: OpenModelSpec) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }

    static func fromCustomDTO(_ dto: CustomHFModelDTO) -> OpenModelSpec {
        let mod: ModelModality
        switch dto.modality {
        case "video_audio": mod = .videoAudio
        case "image": mod = .image
        default: mod = .llmVLM
        }
        let sizeGB = dto.approx_size_gb ?? 0.0
        let arch = (dto.pipeline_arch ?? "gguf").uppercased()
        let badgeZh = sizeGB > 0 ? String(format: "自定义 %@ · %.1f GB", arch, sizeGB) : "自定义 HF · \(arch)"
        let badgeEn = sizeGB > 0 ? String(format: "Custom %@ · %.1f GB", arch, sizeGB) : "Custom HF · \(arch)"
        let patDesc = (dto.include_pattern?.isEmpty == false) ? " (\(dto.include_pattern!))" : ""
        let summaryZh = dto.category ?? "自定义 HuggingFace 仓库 \(dto.repo_id)\(patDesc)"
        let summaryEn = "Custom HuggingFace repo \(dto.repo_id)\(patDesc)"
        let modes: [TaskModeItem]
        let iconName: String
        let defSteps: Double
        switch mod {
        case .videoAudio, .video:
            iconName = "film.stack"
            defSteps = 35
            modes = [
                TaskModeItem(id: "t2va", titleZh: "文生音视频", titleEn: "Text to Video+Audio", subtitleZh: "输入镜头与声音描述生成视频", subtitleEn: "Generate video & audio from prompts", icon: "film"),
                TaskModeItem(id: "fl2va", titleZh: "首尾帧生视频", titleEn: "Keyframes to Video", subtitleZh: "从相册选首帧/尾帧参考图", subtitleEn: "Pick start/end keyframes from Photos", icon: "photo.on.rectangle.angled")
            ]
        case .image:
            iconName = "sparkles.rectangle.stack"
            defSteps = 20
            modes = [
                TaskModeItem(id: "t2i", titleZh: "高清文生图", titleEn: "Text-to-Image", subtitleZh: "自定义 HuggingFace 图像生成管线", subtitleEn: "Custom HuggingFace image generation", icon: "photo.artframe"),
                TaskModeItem(id: "i2i_edit", titleZh: "图生图重绘", titleEn: "Image Restyle", subtitleZh: "从相册选参考图重绘", subtitleEn: "Pick reference photo to restyle", icon: "wand.and.rays")
            ]
        case .llmVLM, .comfyUI:
            iconName = "shippingbox.fill"
            defSteps = 20
            modes = [
                TaskModeItem(id: "chat", titleZh: "深度推理对话", titleEn: "Reasoning Chat", subtitleZh: "自定义 HuggingFace 模型对话与推理", subtitleEn: "Custom HuggingFace model chat & reasoning", icon: "bubble.left.and.bubble.right"),
                TaskModeItem(id: "vlm", titleZh: "看图反推/视觉", titleEn: "Vision & Prompt", subtitleZh: "多模态看图分析与提示词生成", subtitleEn: "Multimodal vision & prompt analysis", icon: "eye"),
                TaskModeItem(id: "api", titleZh: "API 协议网关", titleEn: "API Gateway", subtitleZh: "OpenAI (/v1) 与 Anthropic (/v1/messages) 接口", subtitleEn: "OpenAI (/v1) & Anthropic (/v1/messages) endpoints", icon: "network")
            ]
        }
        return OpenModelSpec(
            id: dto.id,
            nameZh: dto.name,
            nameEn: dto.name,
            shortTitleZh: dto.name,
            shortTitleEn: dto.name,
            repoId: dto.repo_id,
            badgeZh: badgeZh,
            badgeEn: badgeEn,
            summaryZh: summaryZh,
            summaryEn: summaryEn,
            approxSizeGB: sizeGB,
            icon: iconName,
            modality: mod,
            defaultSteps: defSteps,
            modes: modes,
            defaultFiles: [],
            includePattern: dto.include_pattern ?? "",
            isCustom: true
        )
    }
}

let OPEN_MODEL_CATALOG: [OpenModelSpec] = [
    OpenModelSpec(
        id: "minimax_h3",
        nameZh: "MiniMax-H3 (33B)",
        nameEn: "MiniMax-H3 (33B)",
        repoId: "MiniMaxAI/MiniMax-H3",
        badgeZh: "BF16 原生 · 134 GB",
        badgeEn: "Native BF16 · 134 GB",
        summaryZh: "全模态 BF16 原生满血精度同步生成高清视频与 32kHz 立体声音效、人声对白及配乐",
        summaryEn: "Full-modality native BF16 generation of HD video with synchronized 32kHz stereo SFX, dialogue & music",
        approxSizeGB: 134.1,
        icon: "film.stack",
        modality: .videoAudio,
        defaultSteps: 35,
        modes: [
            TaskModeItem(
                id: "t2va",
                titleZh: "文生音视频",
                titleEn: "Text to Video+Audio",
                subtitleZh: "输入镜头与声音描述，同步生成视频与立体声音轨",
                subtitleEn: "Generate HD video and synchronized stereo audio from scene & sound prompts",
                icon: "film"
            ),
            TaskModeItem(
                id: "i2va",
                titleZh: "首帧生视频",
                titleEn: "Start Frame to Video",
                subtitleZh: "只给第一帧，从这张图往后生成画面与声音",
                subtitleEn: "Animate forward from a single start frame, with synced audio",
                icon: "photo.badge.arrow.down"
            ),
            TaskModeItem(
                id: "l2va",
                titleZh: "尾帧生视频",
                titleEn: "End Frame to Video",
                subtitleZh: "只给最后一帧，生成一段收束到这张图的视频",
                subtitleEn: "Generate a clip that lands exactly on a given end frame",
                icon: "photo.badge.checkmark"
            ),
            TaskModeItem(
                id: "fl2va",
                titleZh: "首尾帧生视频",
                titleEn: "First/Last Frame to Video",
                subtitleZh: "同时给首帧和尾帧，锁定起止画面与中间的运镜过渡",
                subtitleEn: "Give both start and end frames to lock composition and the transition",
                icon: "photo.on.rectangle.angled"
            ),
            TaskModeItem(
                id: "ref2va",
                titleZh: "参考生音视频",
                titleEn: "Reference to Video+Audio",
                subtitleZh: "最多 9 张图、3 段视频、3 段音频做参考，提示词里用 <Picture 1> 等标签指代",
                subtitleEn: "Up to 9 images, 3 videos and 3 audio clips as references, cited as <Picture 1> etc.",
                icon: "square.stack.3d.up"
            )
        ],
        defaultFiles: (1...13).map { i in
            ModelFileItem(
                name: String(format: "FL2VA/transformer/model-%05d-of-00013.safetensors", i),
                desc: "DiT 音视频主干 BF16 \(i)/13",
                exists: false,
                size_gb: 0
            )
        } + (1...14).map { i in
            ModelFileItem(
                name: String(format: "FL2VA/text_encoder/model-%05d-of-00014.safetensors", i),
                desc: "多模态编码器 BF16 \(i)/14",
                exists: false,
                size_gb: 0
            )
        } + [
            ModelFileItem(name: "FL2VA/video_vae/source/model.safetensors", desc: "3D 视频 VAE BF16 (9.7 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "FL2VA/audio_vae/model.safetensors", desc: "32kHz 立体声 VAE BF16 (0.6 GB)", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "minimax_h3_image",
        nameZh: "MiniMax-H3-Image",
        nameEn: "MiniMax-H3-Image",
        repoId: "MiniMaxAI/MiniMax-H3-Image",
        badgeZh: "BF16 原生 · 63.9 GB + 共用编码器",
        badgeEn: "Native BF16 · 63.9 GB + shared encoder",
        summaryZh: "MiniMax 官方 BF16 图像模型：6 步出 4K 级图像，支持参考图编辑；文本编码器与 MiniMax-H3 共用（未下载 H3 时另需 62.1 GB）",
        summaryEn: "Official MiniMax BF16 image model: 4K-class images in 6 steps with reference-image editing; shares the MiniMax-H3 text encoder (62.1 GB extra if H3 is absent)",
        approxSizeGB: 126.0,
        icon: "photo.stack",
        modality: .image,
        defaultSteps: 6,
        modes: [
            TaskModeItem(
                id: "t2i",
                titleZh: "4K 文生图",
                titleEn: "4K Text-to-Image",
                subtitleZh: "固定 6 步采样；提示词用详细的英文整图描述效果最好",
                subtitleEn: "Fixed 6-step sampling; a detailed English description of the whole image works best",
                icon: "photo.artframe"
            ),
            TaskModeItem(
                id: "i2i_edit",
                titleZh: "参考图编辑",
                titleEn: "Reference-Image Edit",
                subtitleZh: "上传 1~16 张参考图，提示词里用 <Picture 1>、<Picture 2>… 指代，可保持人物与场景一致",
                subtitleEn: "Upload 1–16 references and refer to them as <Picture 1>, <Picture 2>… to keep characters and scenes consistent",
                icon: "wand.and.rays"
            )
        ],
        defaultFiles: (1...12).map { i in
            ModelFileItem(
                name: String(format: "transformer/model-%05d-of-00012.safetensors", i),
                desc: "图像 DiT 主干 BF16 \(i)/12",
                exists: false,
                size_gb: 0
            )
        } + (1...14).map { i in
            ModelFileItem(
                name: String(format: "text_encoder/model-%05d-of-00014.safetensors", i),
                desc: "多模态编码器 BF16 \(i)/14",
                exists: false,
                size_gb: 0
            )
        } + [
            ModelFileItem(name: "vae/diffusion_pytorch_model.safetensors", desc: "图像 VAE (2.2 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "pdd_fused_heads.safetensors", desc: "6 步蒸馏输出头 (12 MB)", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "comfyui_studio",
        nameZh: "ComfyUI 节点工作台",
        nameEn: "ComfyUI Node Studio",
        repoId: "comfyanonymous/ComfyUI",
        badgeZh: "节点视频创作 · 云盘软链持久化",
        badgeEn: "Node Video Studio · Drive Symlink Persist",
        summaryZh: "一键拉起云端 ComfyUI + Manager 全屏节点画布，自定义节点、视频权重与工作流自动软链持久化至 Google Drive",
        summaryEn: "1-click cloud ComfyUI + Manager fullscreen node canvas with Google Drive symlink persistence for custom nodes, weights & workflows",
        approxSizeGB: 0.0,
        icon: "point.3.connected.trianglepath.dotted",
        modality: .comfyUI,
        defaultSteps: 25,
        modes: [
            TaskModeItem(
                id: "comfy_canvas",
                titleZh: "全屏节点画布",
                titleEn: "Fullscreen Node Canvas",
                subtitleZh: "一键部署并跳转浏览器全屏节点画布，复用 G4 保活、折线监控与自动休眠",
                subtitleEn: "Deploy & open fullscreen canvas in browser with G4 keepalive, live telemetry & auto-sleep",
                icon: "macwindow.badge.plus"
            ),
            TaskModeItem(
                id: "comfy_assets",
                titleZh: "极速下权重与回传成片",
                titleEn: "Fast Weights & Pull Outputs",
                subtitleZh: "8 线程极速下载视频模型至云盘 models 目录，并将 output 成片一键回传 Mac",
                subtitleEn: "8-thread fast model download to Drive models folder & 1-click output pull to Mac",
                icon: "externaldrive.badge.icloud"
            )
        ],
        defaultFiles: []
    ),
    OpenModelSpec(
        id: "qwen38_27b",
        nameZh: "Qwen3.8-27B",
        nameEn: "Qwen3.8-27B",
        repoId: "Qwen/Qwen3.8-27B",
        badgeZh: "BF16 原生 · 51.8 GB",
        badgeEn: "Native BF16 · 51.8 GB",
        summaryZh: "Hugging Face 榜首开源多模态大模型（BF16 原生满血精度，支持最高 256K 上下文，需大显存实例）",
        summaryEn: "Top Hugging Face open multimodal LLM (full native BF16 precision, up to 256K context, high-VRAM GPU)",
        approxSizeGB: 51.8,
        icon: "brain",
        modality: .llmVLM,
        defaultSteps: 20,
        modes: [
            TaskModeItem(
                id: "chat",
                titleZh: "深度推理对话",
                titleEn: "Deep Reasoning Chat",
                subtitleZh: "支持长思维链推理、代码架构、复杂逻辑与分镜脚本创作",
                subtitleEn: "Long chain-of-thought reasoning, software architecture, logic & storyboard scripting",
                icon: "bubble.left.and.bubble.right"
            ),
            TaskModeItem(
                id: "vlm",
                titleZh: "图像视觉理解",
                titleEn: "Vision Understanding",
                subtitleZh: "拖入图片进行画面解析、OCR 识别或反推专业生图/生视频提示词",
                subtitleEn: "Drop images for scene analysis, OCR, or reverse-engineering image/video prompts",
                icon: "eye"
            ),
            TaskModeItem(
                id: "api",
                titleZh: "API 协议网关",
                titleEn: "API Protocol Gateway",
                subtitleZh: "提供 OpenAI (/v1) 与 Anthropic (/v1/messages) 双协议公网接口",
                subtitleEn: "Public dual-protocol gateway compatible with OpenAI (/v1) & Anthropic (/v1/messages)",
                icon: "network"
            )
        ],
        defaultFiles: (1...18).map { i in
            ModelFileItem(
                name: String(format: "model-%05d-of-00018.safetensors", i),
                desc: "权重分片 \(i)/18",
                exists: false,
                size_gb: 0
            )
        }
    ),
    OpenModelSpec(
        id: "qwen38_27b_gguf",
        nameZh: "Qwen3.8-27B-GGUF",
        nameEn: "Qwen3.8-27B-GGUF",
        repoId: "unsloth/Qwen3.8-27B-GGUF",
        badgeZh: "UD-Q4_K_M + MTP · 18.7 GB",
        badgeEn: "UD-Q4_K_M + MTP · 18.7 GB",
        summaryZh: "Unsloth 动态 4-bit 量化 + 原生 MTP 多词元投机解码加速，支持最高 256K 上下文与多模态视觉理解",
        summaryEn: "Unsloth dynamic 4-bit quantization + native MTP speculative decoding (~100 tok/s), up to 256K context & vision",
        approxSizeGB: 18.7,
        icon: "cpu",
        modality: .llmVLM,
        defaultSteps: 20,
        modes: [
            TaskModeItem(
                id: "chat",
                titleZh: "深度推理对话",
                titleEn: "Deep Reasoning Chat",
                subtitleZh: "UD-Q4_K_M + MTP 投机解码极速推理，支持长思维链、代码架构与超长上下文分析",
                subtitleEn: "Ultra-fast UD-Q4_K_M + MTP speculative decoding for CoT reasoning, coding & long-context analysis",
                icon: "bubble.left.and.bubble.right"
            ),
            TaskModeItem(
                id: "vlm",
                titleZh: "图像视觉理解",
                titleEn: "Vision Understanding",
                subtitleZh: "内置 mmproj-F16 视觉投影器，支持图片解析、OCR 与生图/生视频提示词反推",
                subtitleEn: "Built-in mmproj-F16 vision projector for image analysis, OCR & prompt reverse-engineering",
                icon: "eye"
            ),
            TaskModeItem(
                id: "api",
                titleZh: "API 协议网关",
                titleEn: "API Protocol Gateway",
                subtitleZh: "提供 OpenAI (/v1) 与 Anthropic (/v1/messages) 双协议公网接口",
                subtitleEn: "Public dual-protocol gateway compatible with OpenAI (/v1) & Anthropic (/v1/messages)",
                icon: "network"
            )
        ],
        defaultFiles: [
            ModelFileItem(name: "Qwen3.8-27B-UD-Q4_K_M.gguf", desc: "Unsloth UD-Q4_K_M 主模型 (16.5 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "mmproj-F16.gguf", desc: "多模态视觉投影器 (0.93 GB)", exists: false, size_gb: 0),
            ModelFileItem(name: "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", desc: "Unsloth MTP 多词元投机解码头 (1.31 GB)", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "qwen_image_21",
        nameZh: "Qwen-Image-2.1",
        nameEn: "Qwen-Image-2.1",
        repoId: "Qwen/Qwen-Image-2.1",
        badgeZh: "文生图与修图 · 30.9 GB",
        badgeEn: "Text-to-Image & Edit · 30.9 GB",
        summaryZh: "通义千问新一代旗舰图像生成与自然语言多图指令编辑模型",
        summaryEn: "Qwen flagship image generation & natural-language multi-image instruction editing model",
        approxSizeGB: 30.9,
        icon: "paintbrush.pointed",
        modality: .image,
        defaultSteps: 40,
        modes: [
            TaskModeItem(
                id: "t2i",
                titleZh: "高清文生图",
                titleEn: "HD Text-to-Image",
                subtitleZh: "复杂构图、精准中英文排版与高质感摄影/插画生成",
                subtitleEn: "Complex composition, accurate bilingual typography & high-fidelity photography/illustration",
                icon: "photo.artframe"
            ),
            TaskModeItem(
                id: "i2i_edit",
                titleZh: "多图指令修图",
                titleEn: "Multi-Image Instruction Edit",
                subtitleZh: "上传 1~2 张原图，用自然语言指令改局部、换背景或融合主体",
                subtitleEn: "Upload 1–2 source images to edit regions, swap backgrounds, or fuse subjects via natural language",
                icon: "wand.and.rays"
            )
        ],
        defaultFiles: [
            ModelFileItem(name: "transformer/diffusion_pytorch_model-00001-of-00002.safetensors", desc: "DiT 1/2", exists: false, size_gb: 0),
            ModelFileItem(name: "transformer/diffusion_pytorch_model-00002-of-00002.safetensors", desc: "DiT 2/2", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00001-of-00004.safetensors", desc: "编码器 1/4", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00002-of-00004.safetensors", desc: "编码器 2/4", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00003-of-00004.safetensors", desc: "编码器 3/4", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00004-of-00004.safetensors", desc: "编码器 4/4", exists: false, size_gb: 0),
            ModelFileItem(name: "vae/diffusion_pytorch_model.safetensors", desc: "图像 VAE", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: "z_image_turbo",
        nameZh: "Z-Image-Turbo",
        nameEn: "Z-Image-Turbo",
        repoId: "Tongyi-MAI/Z-Image-Turbo",
        badgeZh: "8步极速生图 · 30.6 GB",
        badgeEn: "8-Step Turbo Image · 30.6 GB",
        summaryZh: "Hugging Face 高人气 8 步蒸馏极速写实摄影与中英文海报生图模型",
        summaryEn: "Popular Hugging Face 8-step distilled model for ultra-fast photorealism & bilingual poster design",
        approxSizeGB: 30.6,
        icon: "bolt.square",
        modality: .image,
        defaultSteps: 8,
        modes: [
            TaskModeItem(
                id: "t2i",
                titleZh: "8 步极速文生图",
                titleEn: "8-Step Fast Text-to-Image",
                subtitleZh: "仅需 8 步采样，秒级输出商业级人像摄影与中英文排版画面",
                subtitleEn: "Commercial-grade portrait photography & bilingual poster typography in just 8 steps",
                icon: "bolt"
            ),
            TaskModeItem(
                id: "i2i_edit",
                titleZh: "图生图风格重绘",
                titleEn: "Image-to-Image Restyle",
                subtitleZh: "上传参考底图并调节重绘幅度，保留构图生成全新细节",
                subtitleEn: "Upload a reference image and adjust denoising strength to restyle while preserving layout",
                icon: "square.on.square"
            )
        ],
        defaultFiles: [
            ModelFileItem(name: "transformer/diffusion_pytorch_model-00001-of-00003.safetensors", desc: "DiT 1/3", exists: false, size_gb: 0),
            ModelFileItem(name: "transformer/diffusion_pytorch_model-00002-of-00003.safetensors", desc: "DiT 2/3", exists: false, size_gb: 0),
            ModelFileItem(name: "transformer/diffusion_pytorch_model-00003-of-00003.safetensors", desc: "DiT 3/3", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00001-of-00003.safetensors", desc: "编码器 1/3", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00002-of-00003.safetensors", desc: "编码器 2/3", exists: false, size_gb: 0),
            ModelFileItem(name: "text_encoder/model-00003-of-00003.safetensors", desc: "编码器 3/3", exists: false, size_gb: 0),
            ModelFileItem(name: "vae/diffusion_pytorch_model.safetensors", desc: "图像 VAE", exists: false, size_gb: 0)
        ]
    ),
    OpenModelSpec(
        id: GPT_IMAGE_MODEL_ID,
        nameZh: "GPT-Image",
        nameEn: "GPT-Image",
        repoId: "openai/gpt-image",
        badgeZh: "OpenAI API 直连 · 无需实例",
        badgeEn: "OpenAI API Direct · No Runtime",
        summaryZh: "由本机直连 OpenAI 图像接口出图与编辑，不占用 Colab 实例；按 OpenAI 的 API 计费，需自备 API Key",
        summaryEn: "Generates and edits images through the OpenAI Images API straight from this device; no Colab runtime, billed by OpenAI, bring your own API key",
        approxSizeGB: 0.0,
        icon: "cloud",
        modality: .image,
        defaultSteps: 1,
        modes: [
            TaskModeItem(
                id: "t2i",
                titleZh: "文生图",
                titleEn: "Text-to-Image",
                subtitleZh: "可指定尺寸与质量档位，最高到 3840×2160",
                subtitleEn: "Choose size and quality, up to 3840×2160",
                icon: "photo.artframe"
            ),
            TaskModeItem(
                id: "i2i_edit",
                titleZh: "参考图与图像编辑",
                titleEn: "Reference & Image Edit",
                subtitleZh: "上传 1~16 张参考图，用自然语言改图、换背景或融合多张图里的主体",
                subtitleEn: "Upload 1–16 references to edit, swap backgrounds, or combine subjects via natural language",
                icon: "wand.and.rays"
            )
        ],
        defaultFiles: []
    )
]

struct ResolutionOption: Identifiable, Hashable {
    let id: String
    let labelZh: String
    let labelEn: String
    let width: Int
    let height: Int

    var label: String { L(labelZh, labelEn) }

    static func == (lhs: ResolutionOption, rhs: ResolutionOption) -> Bool {
        lhs.id == rhs.id && lhs.width == rhs.width && lhs.height == rhs.height
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(width)
        hasher.combine(height)
    }
}

let VIDEO_RESOLUTIONS: [ResolutionOption] = [
    ResolutionOption(id: "v_832_480", labelZh: "832 × 480 (16:9 横屏标准 · 推荐)", labelEn: "832 × 480 (16:9 Landscape · Recommended)", width: 832, height: 480),
    ResolutionOption(id: "v_480_832", labelZh: "480 × 832 (9:16 竖屏短视频)", labelEn: "480 × 832 (9:16 Vertical Short)", width: 480, height: 832),
    ResolutionOption(id: "v_1280_720", labelZh: "1280 × 720 (16:9 720P 高清)", labelEn: "1280 × 720 (16:9 720P HD)", width: 1280, height: 720),
    ResolutionOption(id: "v_720_1280", labelZh: "720 × 1280 (9:16 720P 竖屏)", labelEn: "720 × 1280 (9:16 720P Vertical)", width: 720, height: 1280),
    ResolutionOption(id: "v_640_640", labelZh: "640 × 640 (1:1 方形)", labelEn: "640 × 640 (1:1 Square)", width: 640, height: 640)
]

let IMAGE_RESOLUTIONS: [ResolutionOption] = [
    ResolutionOption(id: "i_1024_1024", labelZh: "1024 × 1024 (1:1 超清方图 · 推荐)", labelEn: "1024 × 1024 (1:1 Square HD · Recommended)", width: 1024, height: 1024),
    ResolutionOption(id: "i_1344_768", labelZh: "1344 × 768 (16:9 电影横幅)", labelEn: "1344 × 768 (16:9 Cinematic Widescreen)", width: 1344, height: 768),
    ResolutionOption(id: "i_768_1344", labelZh: "768 × 1344 (9:16 竖版海报)", labelEn: "768 × 1344 (9:16 Vertical Poster)", width: 768, height: 1344),
    ResolutionOption(id: "i_1536_1024", labelZh: "1536 × 1024 (3:2 单反摄影)", labelEn: "1536 × 1024 (3:2 DSLR Photo)", width: 1536, height: 1024),
    ResolutionOption(id: "i_2048_2048", labelZh: "2048 × 2048 (1:1 2K 方图)", labelEn: "2048 × 2048 (1:1 2K Square)", width: 2048, height: 2048),
    ResolutionOption(id: "i_2048_1152", labelZh: "2048 × 1152 (16:9 2K 横幅)", labelEn: "2048 × 1152 (16:9 2K Widescreen)", width: 2048, height: 1152),
    ResolutionOption(id: "i_1152_2048", labelZh: "1152 × 2048 (9:16 2K 竖版)", labelEn: "1152 × 2048 (9:16 2K Vertical)", width: 1152, height: 2048)
]

// MiniMax-H3-Image 的有效范围约 4~20MP，低于 4MP 的请求会在更大画布上生成再缩小，所以最小档就是 4MP（比 GPT-Image 的 2K 档略大）
let H3_IMAGE_ASPECTS: [(id: String, w: Int, h: Int)] = [
    ("1:1", 1, 1), ("4:3", 4, 3), ("3:2", 3, 2), ("16:9", 16, 9), ("21:9", 21, 9),
    ("4:5", 4, 5), ("3:4", 3, 4), ("2:3", 2, 3), ("9:16", 9, 16), ("9:21", 9, 21)
]
let H3_IMAGE_TIERS: [(id: String, label: String, pixels: Int)] = [
    ("4MP", "2K · 4MP", 2048 * 2048), ("8MP", "4K · 8MP", 2 * 2048 * 2048),
    ("12MP", "12MP", 3 * 2048 * 2048), ("16MP", "16MP", 4096 * 4096)
]

/// 宽、高各自取最接近的 32 的倍数（与 ComfyUI 的比例选择器同一算法）
func h3ImageSize(aspect: String, tier: String) -> (width: Int, height: Int) {
    let a = H3_IMAGE_ASPECTS.first { $0.id == aspect } ?? H3_IMAGE_ASPECTS[0]
    let pixels = Double((H3_IMAGE_TIERS.first { $0.id == tier } ?? H3_IMAGE_TIERS[0]).pixels)
    func snap(_ v: Double) -> Int { Int((v / 32).rounded()) * 32 }
    return (snap((pixels * Double(a.w) / Double(a.h)).squareRoot()), snap((pixels * Double(a.h) / Double(a.w)).squareRoot()))
}

struct PresetTemplate: Identifiable {
    let id: String
    let category: ModelModality
    let nameZh: String
    let nameEn: String
    let visualZh: String
    let visualEn: String
    let soundscape: String
    let music: String

    var name: String { L(nameZh, nameEn) }
    var visual: String { L(visualZh, visualEn) }
}

let PRESET_TEMPLATES: [PresetTemplate] = [
    PresetTemplate(
        id: "tpl_shinkai_train",
        category: .videoAudio,
        nameZh: "新海诚风 · 盛夏樱花铁道口",
        nameEn: "Makoto Shinkai · Cherry Blossom Railway",
        visualZh: "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        visualEn: "Masterpiece Makoto Shinkai anime style, 2D cel-shaded animation. A teenage girl in a Japanese sailor school uniform standing at a sunlit railway crossing in late spring. A green commuter train speeds past in front of her, creating a strong gust of wind that blows her long black hair and a flurry of pink cherry blossom petals across the frame. Dramatic lens flare, glowing cumulous clouds in a deep blue sky, vibrant cinematic lighting, smooth 24fps keyframe animation.",
        soundscape: "Metallic clacking of train wheels on railway tracks, rhythmic dinging of the railroad crossing warning bell, warm spring wind rustling through leaves, a soft gasp from the girl.",
        music: "Emotional and nostalgic solo piano melody joined by warm acoustic guitar and soaring cinematic strings."
    ),
    PresetTemplate(
        id: "tpl_ufotable_duel",
        category: .videoAudio,
        nameZh: "飞碟社风 · 苍蓝雷光剑术对决",
        nameEn: "Ufotable Sakuga · Azure Lightning Duel",
        visualZh: "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        visualEn: "High-budget ufotable anime action sakuga, dynamic fast-paced camera tracking. A silver-haired anime swordsman in a dark haori coat dashes across a shattered stone temple under a blood-red full moon, swinging a katana wreathed in crackling blue lightning and swirling water dragon effects. Sparks and glowing embers fly toward the camera in slow-motion impact frames, intense anime shading, dramatic motion blur.",
        soundscape: "Sharp metallic sword unsheathing and clashing clang, roaring thunder and crackling electric arcs, heavy stone debris crumbling, fierce battle cry.",
        music: "Epic dark orchestral battle music with fast taiko drums, intense choir, and distorted electric guitar riffs."
    ),
    PresetTemplate(
        id: "tpl_iceland_fpv",
        category: .videoAudio,
        nameZh: "冰岛巨型瀑布 FPV 穿越机航拍",
        nameEn: "Iceland Waterfall FPV Drone Dive",
        visualZh: "FPV drone shot diving down a majestic Icelandic waterfall surrounded by lush green moss and basalt columns under golden hour sunlight, mist rising into a rainbow.",
        visualEn: "FPV drone shot diving down a majestic Icelandic waterfall surrounded by lush green moss and basalt columns under golden hour sunlight, mist rising into a rainbow.",
        soundscape: "Roaring waterfall rushing water, wind whooshing past the camera, distant seabirds calling.",
        music: "Epic cinematic acoustic guitar and soaring cello."
    ),
    PresetTemplate(
        id: "tpl_cyber_poster",
        category: .image,
        nameZh: "东方赛博霓虹海报（含中文排版）",
        nameEn: "Cyberpunk Neon Bund Poster",
        visualZh: "超清商业电影海报摄影，雨夜霓虹闪烁的未来上海外滩街道，一位身着机能风银色夹克的少女回眸凝视镜头，景深虚化光斑，顶部醒目发光现代中文字体排版写着“未来纪元 2026”，极致光影细节，8K画质。",
        visualEn: "Ultra-HD commercial sci-fi movie poster photography, rain-slicked cyberpunk Shanghai Bund street glowing with neon reflections at night, a young woman in a futuristic silver techwear jacket looking back at the camera, shallow depth of field bokeh flares, bold glowing modern title typography at the top reading \"FUTURE ERA 2026\" (未来纪元 2026), intricate lighting details, 8K.",
        soundscape: "",
        music: ""
    ),
    PresetTemplate(
        id: "tpl_film_cafe",
        category: .image,
        nameZh: "哈苏胶片感 · 午后东京咖啡馆",
        nameEn: "Hasselblad Film · Afternoon Cafe",
        visualZh: "Hasselblad medium format film photography, soft golden afternoon sunlight streaming through wooden blinds onto a cozy minimalist Tokyo cafe table, a ceramic pour-over coffee dripper with rising steam beside an open sketchbook, natural film grain, ultra-detailed textures.",
        visualEn: "Hasselblad medium format film photography, soft golden afternoon sunlight streaming through wooden blinds onto a cozy minimalist Tokyo cafe table, a ceramic pour-over coffee dripper with rising steam beside an open sketchbook, natural film grain, ultra-detailed textures.",
        soundscape: "",
        music: ""
    ),
    PresetTemplate(
        id: "tpl_llm_storyboard",
        category: .llmVLM,
        nameZh: "生成 3 组新海诚风 AI 视频分镜",
        nameEn: "Design 3 Shinkai AI Video Storyboards",
        visualZh: "请帮我构思 3 组极具画面感的新海诚风格动画短片分镜，每组包含：1) 英文 Visual & Camera 镜头提示词；2) Soundscape 环境音描述；3) Background Music 配乐描述。",
        visualEn: "Please design 3 vivid Makoto Shinkai-style anime short film storyboard shots. For each shot, provide: 1) Visual & Camera prompt; 2) Soundscape & SFX prompt; 3) Background Music prompt.",
        soundscape: "",
        music: ""
    ),
    PresetTemplate(
        id: "tpl_vlm_reverse",
        category: .llmVLM,
        nameZh: "看图深度解析并反推生图/生视频 Prompt",
        nameEn: "Analyze Photo & Reverse-Engineer Prompt",
        visualZh: "请仔细观察这张图片的主体、光影色调、镜头景别与艺术风格，先做简要中文分析，然后反推出一段高质量的英文生图/生视频 Prompt。",
        visualEn: "Carefully analyze the subject, lighting, color palette, camera framing, and artistic style of this image. First give a concise visual breakdown, then write a high-quality English image/video generation prompt.",
        soundscape: "",
        music: ""
    )
]

// MARK: - Server API Wire DTOs

struct LoadedModelSummary: Codable, Identifiable, Hashable {
    var id: String
    var name: String
    var mode: String?
    var vram_gb: Double?
    var context_length: Int?

    var localizedName: String {
        if let spec = OPEN_MODEL_CATALOG.first(where: { $0.id == id }) {
            return spec.name
        }
        return localizeServerText(name)
    }
}

struct ComfyOutputMediaItem: Codable, Identifiable, Hashable {
    var id: String { rel_path }
    var rel_path: String
    var filename: String
    var media_type: String?
    var size_mb: Double?
    var mtime: Int?
}

struct ComfyPresetPackItem: Codable, Identifiable, Hashable {
    var id: String
    var title: String
    var badge: String
    var scenario: String
    var approx_size_gb: Double
    var workflows: [String]
    var installed_models: Int
    var total_models: Int
    var workflows_installed: Bool
    var ready: Bool
    /// 服务端算好的：本包（含工作流要用的别包文件）还缺多少 GB、实例盘上还剩多少 GB；旧服务端没有这两个字段
    var missing_gb: Double? = nil
    var disk_free_gb: Double? = nil

    var localizedTitle: String {
        guard LanguageManager.shared.isEN else { return title }
        switch id {
        case "wan22_i2v_flf2v":
            return "Wan2.2 14B · Image-to-Video + First/Last Frame Dual Workflows"
        case "minimax_h3_i2v_flf2v":
            return "MiniMax-H3 Turbo · Image-to-Video+Audio + First/Last Frame"
        case "ltx23_flf2v":
            return "LTX-2.3 (22B) · Cinematic Camera Motion First/Last Frame + Audio"
        case "hunyuan15_i2v_1080p":
            return "HunyuanVideo 1.5 · Single-Image Direct 1080P Super-Res Workflow"
        default:
            return localizeServerText(title)
        }
    }

    var localizedBadge: String {
        guard LanguageManager.shared.isEN else { return badge }
        switch id {
        case "wan22_i2v_flf2v":
            return "Beginner Pick · 4-Step Fast"
        case "minimax_h3_i2v_flf2v":
            return "Stereo Audio · 8-Step Turbo"
        case "ltx23_flf2v":
            return "Cinematic Camera · Only 2 Weights"
        case "hunyuan15_i2v_1080p":
            return "720P + 1080P Dual Output"
        default:
            return localizeServerText(badge)
        }
    }

    var localizedScenario: String {
        guard LanguageManager.shared.isEN else { return scenario }
        switch id {
        case "wan22_i2v_flf2v":
            return "Includes 'Single Image to Video' & 'First/Last Frame Smooth Transition' templates (~25s on G4, 100% built-in nodes, zero missing-node errors)"
        case "minimax_h3_i2v_flf2v":
            return "Includes single-image and first/last-frame templates generating HD video with synchronized stereo SFX, dialogue & music in one pass"
        case "ltx23_flf2v":
            return "Optimized for dramatic camera moves between start & end keyframes (e.g. aerial crane dive to low-angle close-up + synthesized soundtrack)"
        case "hunyuan15_i2v_1080p":
            return "Latent super-resolution pipeline enabled by default — upload 1 image to render both 720P base video and 1920×1080 Full HD output"
        default:
            return localizeServerText(scenario)
        }
    }

    static func localizedWorkflowLabel(_ rawName: String) -> String {
        let stripped = rawName.replacingOccurrences(of: ".json", with: "")
        guard LanguageManager.shared.isEN else { return stripped }
        let map: [String: String] = [
            "01_Wan2.2_极速单图生视频(4步傻瓜版)": "01_Wan2.2_Fast_Single_I2V_4Step",
            "02_Wan2.2_首尾帧丝滑过渡(双图4步极速)": "02_Wan2.2_FirstLastFrame_4Step",
            "03_MiniMax-H3_极速图生有声视频(单图带立体声)": "03_MiniMax-H3_Fast_I2VA_Stereo",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声)": "04_MiniMax-H3_FirstLastFrame_Stereo",
            "05_LTX-2.3_首尾帧电影运镜(双图极速带音效)": "05_LTX-2.3_Cinematic_FLF2V_Audio",
            "06_Hunyuan1.5_单图直出1080P超清视频": "06_Hunyuan1.5_Direct_1080P_I2V"
        ]
        return map[stripped] ?? stripped
    }
}

let DEFAULT_COMFY_PRESET_PACKS: [ComfyPresetPackItem] = [   // 离线默认清单：与 app.py 的 COMFY_PRESET_WORKFLOW_PACKS 保持一致，连上实例后以服务端列表为准
    ComfyPresetPackItem(
        id: "wan22_i2v_flf2v",
        title: "万相 Wan2.2 14B · 图生视频 + 首尾帧双工作流",
        badge: "新手首选 · 4步极速",
        scenario: "含「单图动起来」与「首尾双图丝滑过渡」2 套中文画布，G4 约 25 秒出片，纯官方内置节点零红框报错",
        approx_size_gb: 35.4,
        workflows: [
            "01_Wan2.2_极速单图生视频(4步傻瓜版).json",
            "02_Wan2.2_首尾帧丝滑过渡(双图4步极速).json"
        ],
        installed_models: 0,
        total_models: 6,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "minimax_h3_i2v_flf2v",
        title: "MiniMax-H3 Turbo · 图生有声视频 + 首尾帧双工作流",
        badge: "自带立体声音效 · 8步加速",
        scenario: "含「单图生有声视频」与「首尾双图有声过渡」2 套极简画布，一次生成高清动态画面 + 同步环境音效/配乐/对白",
        approx_size_gb: 53.7,
        workflows: [
            "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json"
        ],
        installed_models: 0,
        total_models: 6,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "minimax_h3_hq_int8",
        title: "MiniMax-H3 附加 · 不剪枝 int8 主干（首尾帧 + 参考主体）+ 官方 / 无审查 int8 编码器",
        badge: "自动补齐基础包文件 · 可常驻显存",
        scenario: "给 03/04 与导演台工作流的下拉框多四个选项：不剪枝 int8 的 fl2va / ref2va 主干，以及官方与 Heretic 两个 int8 编码器，本包约 113 GB；连同 03/04 工作流默认要用的基础包文件合计约 152 GB，不要和满血 bf16 包同时装",
        approx_size_gb: 152.37,
        workflows: [
            "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json"
        ],
        installed_models: 0,
        total_models: 9,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "minimax_h3_hq_bf16",
        title: "MiniMax-H3 附加 · 满血 bf16 主干 + 官方 bf16 编码器 + fp16 视频 VAE",
        badge: "自动补齐基础包文件 · 最高精度",
        scenario: "给 03/04 工作流的下拉框多两个选项：满血 bf16 主干、官方 bf16 编码器与 fp16 视频 VAE，本包约 115 GB；连同 03/04 工作流默认要用的基础包文件合计约 154 GB，不要再同时装其它大包",
        approx_size_gb: 153.69,
        workflows: [
            "03_MiniMax-H3_极速图生有声视频(单图带立体声).json",
            "04_MiniMax-H3_首尾帧有声过渡(双图带立体声).json"
        ],
        installed_models: 0,
        total_models: 8,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "minimax_h3_director",
        title: "MiniMax-H3 导演台 · 多段 + 参考主体 + 改视频 + 语义桥 / 二采放大 / 修脸",
        badge: "自动补齐基础包文件 · 导演台全功能",
        scenario: "H3 Director 插件的 9 套示例工作流 + Comfyit 资源包里的外接参数版与官方 i2v/r2v/t2v 模板共 5 套，以及全部可选权重：参考主体 ref2va 主干（int8 与 w4a8 两种）、fp16 视频 VAE、4 步 Turbo LoRA、语义桥（原版 + BUNNY V1/V2）、3D 潜空间放大（bf16 / fp16 / fp32）、UltraSharp 与 RealESRGAN 像素放大、TAE 实时预览、修脸检测，本包约 41 GB；连同工作流要用的基础包编码器、fl2va 主干、音频 VAE 与 8 步 LoRA 合计约 77 GB",
        approx_size_gb: 77.21,
        workflows: [
            "10_H3导演台_首尾帧多段(fl2v).json",
            "11_H3导演台_文生多段(t2v).json",
            "12_H3导演台_参考主体素材组(r2v).json",
            "13_H3导演台_源视频改视频(v2v).json",
            "14_H3导演台_源视频加参考(rv2v).json",
            "15_H3导演台_外部多组接线(i2v).json",
            "16_H3导演台_外部多组接线(r2v).json",
            "17_H3导演台_二采放大(Refine).json",
            "18_H3导演台_加速版(SageAttention).json",
            "21_H3导演台_外接参数版(i2v).json",
            "22_H3导演台_外接参数版(r2v).json",
            "23_H3官方模板_图生有声视频(i2v).json",
            "24_H3官方模板_参考主体(r2v).json",
            "25_H3官方模板_文生有声视频(t2v).json"
        ],
        installed_models: 0,
        total_models: 18,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "minimax_h3_director_plus",
        title: "MiniMax-H3 导演台扩展 · 全能工作流（混合主干 + Pika 六步加速）",
        badge: "自动补齐基础包与导演台文件",
        scenario: "Comfyit 资源包的两套导演台全能工作流（一采 + SelfLift + 二采放大 + 修脸一条龙）；另下 fl2va/ref2va 混合 int8 主干与 Pika H3 Flex 的 fl2va / ref2va 两个 LoRA，本包约 23 GB；连同工作流要用的基础包与导演台文件合计约 87 GB",
        approx_size_gb: 86.58,
        workflows: [
            "19_H3导演台_全能工作流(混合主干).json",
            "20_H3导演台_全能工作流(Pika六步加速).json"
        ],
        installed_models: 0,
        total_models: 12,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "minimax_h3_director_bf16",
        title: "MiniMax-H3 导演台 · 满血 bf16（参考主体）",
        badge: "满血主干 · 参考主体 r2v",
        scenario: "导演台 4 套参考主体（r2v）工作流的满血版：主干 / 编码器下拉默认是 bf16 ref2va 与官方 bf16 编码器，另含 fp16 视频 VAE、音频 VAE、TAE 实时预览、3D 潜空间放大与修脸检测，合计约 116 GB；盘只够单独装它，不要和满血 bf16 首尾帧包或导演台 int8 包同时装",
        approx_size_gb: 115.82,
        workflows: [
            "12_H3导演台_参考主体素材组(r2v)_满血bf16.json",
            "16_H3导演台_外部多组接线(r2v)_满血bf16.json",
            "22_H3导演台_外接参数版(r2v)_满血bf16.json",
            "24_H3官方模板_参考主体(r2v)_满血bf16.json"
        ],
        installed_models: 0,
        total_models: 7,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "minimax_h3_image_t2i_edit",
        title: "MiniMax-H3-Image · 文生图 + 参考图改图 + 多参考图合成",
        badge: "6步出图 · 多张参考图",
        scenario: "含「文生图」「按原图尺寸改图」「多张参考图合成新画面」3 套画布；主干是官方剪枝 bf16 版，要满血版请用左侧的 MiniMax-H3-Image 模型",
        approx_size_gb: 70.7,
        workflows: [
            "07_MiniMax-H3-Image_文生图(4MP高清).json",
            "08_MiniMax-H3-Image_参考图改图(按原图尺寸).json",
            "09_MiniMax-H3-Image_多参考图合成新画面(16比9).json"
        ],
        installed_models: 0,
        total_models: 4,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "ltx23_flf2v",
        title: "LTX-2.3 (22B) · 首尾帧电影级运镜有声工作流",
        badge: "电影推拉摇移 · 仅 2 个权重",
        scenario: "专为首尾关键帧大幅度镜头运动优化（如高空俯拍平滑推镜至低角度特写 + 自动合成配乐音轨）",
        approx_size_gb: 36.3,
        workflows: [
            "05_LTX-2.3_首尾帧电影运镜(双图极速带音效).json"
        ],
        installed_models: 0,
        total_models: 2,
        workflows_installed: false,
        ready: false
    ),
    ComfyPresetPackItem(
        id: "hunyuan15_i2v_1080p",
        title: "混元 HunyuanVideo 1.5 · 单图直出 1080P 超分工作流",
        badge: "720P + 1080P 双路超清",
        scenario: "已默认激活潜空间超分管线，传 1 张图一键自动渲染 720P 原片与 1920×1080 超高清成片",
        approx_size_gb: 43.5,
        workflows: [
            "06_Hunyuan1.5_单图直出1080P超清视频.json"
        ],
        installed_models: 0,
        total_models: 7,
        workflows_installed: false,
        ready: false
    )
]

struct ComfyRemoteStatus: Codable {
    var is_running: Bool?
    var is_starting: Bool?
    var is_syncing_drive: Bool?
    var is_downloading_weight: Bool?
    var installing_pack_id: String?
    var progress: Double?
    var status_message: String?
    var public_url: String?
    var local_port: Int?
    var queue_running: Int?
    var queue_pending: Int?
    var persistence_mode: String?
    var persistence_label: String?
    var drive_persisted: Bool?
    var persist_root: String?
    var custom_nodes_count: Int?
    var models_count: Int?
    var workflows_count: Int?
    var last_sync_ts: Int?
    var recent_outputs: [ComfyOutputMediaItem]?
    var preset_packs: [ComfyPresetPackItem]?
}

struct ColabStatusResponse: Codable {
    var ok: Bool?
    var cuda_available: Bool?
    var gpu_name: String?
    var vram_free_gb: Double?
    var vram_total_gb: Double?
    var vram_used_gb: Double?
    var model_loaded: Bool?
    var any_model_loaded: Bool?
    var selected_model_id: String?
    var loaded_model_id: String?
    var loaded_model_ids: [String]?
    var loaded_models_detail: [LoadedModelSummary]?
    var current_model: String?
    var message: String?
    var context_length: Int?
    var loaded_context_length: Int?
    var files: [ModelFileItem]?
    var downloaded_count: Int?
    var total_count: Int?
    var all_downloaded: Bool?
    var total_size_gb: Double?
    var drive_persisted: Bool?
    var bf16_shards_count: Int?
    var is_downloading: Bool?
    var is_loading_vram: Bool?
    var task_progress: Double?
    var task_message: String?
    var uptime_seconds: Int?
    var cu_burn_rate_per_hour: Double?
    var session_cu_used: Double?
    var idle_timeout_minutes: Int?
    var idle_remaining_seconds: Int?
    var api_key_enabled: Bool?
    var api_request_count: Int?
    var console_lines: [String]?
    var ram_total_gb: Double?
    var ram_used_gb: Double?
    var ram_free_gb: Double?
    var disk_total_gb: Double?
    var disk_used_gb: Double?
    var disk_free_gb: Double?
    var shm_total_gb: Double?
    var shm_used_gb: Double?
    var shm_free_gb: Double?
    var drive_total_gb: Double?
    var drive_used_gb: Double?
    var drive_free_gb: Double?
    var comfyui: ComfyRemoteStatus?
    var custom_models: [CustomHFModelDTO]?
    var registered_model: CustomHFModelDTO?
    var action_ok: Bool?
    var action_message: String?
}

struct CompletedMediaItem: Codable, Hashable {
    var index: Int?
    var seed: Int?
    var filename: String
    var video_url: String
    var media_type: String?
    var elapsed: Double?
}

struct LiveTelemetrySnapshot: Codable {
    var cuda_available: Bool?
    var gpu_name: String?
    var vram_free_gb: Double?
    var vram_total_gb: Double?
    var vram_used_gb: Double?
    var cu_burn_rate_per_hour: Double?
    var ram_total_gb: Double?
    var ram_used_gb: Double?
    var ram_free_gb: Double?
    var disk_total_gb: Double?
    var disk_used_gb: Double?
    var disk_free_gb: Double?
    var shm_total_gb: Double?
    var shm_used_gb: Double?
    var shm_free_gb: Double?
    var drive_total_gb: Double?
    var drive_used_gb: Double?
    var drive_free_gb: Double?
}

struct ColabJobResponse: Codable {
    var job_id: String
    var model_id: String?
    var status: String
    var progress: Double
    var message: String
    var seed: Int?
    var elapsed: Double?
    var filename: String?
    var video_url: String?
    var media_type: String?
    var text_output: String?
    var thinking_output: String?
    var batch_count: Int?
    var completed_count: Int?
    var completed_videos: [CompletedMediaItem]?
    var console_lines: [String]?
    var telemetry: LiveTelemetrySnapshot?
}

struct LocalMediaItem: Identifiable, Hashable {
    var id: String { filename }
    let url: URL
    let filename: String
    let isImage: Bool
    let createdAt: Date
    let elapsedSeconds: Double?
    let fileSizeMB: Double
}

struct HFTrendingModelItem: Identifiable, Codable, Hashable {
    var id: String { repo_id }
    let repo_id: String
    let name: String
    let pipeline_tag: String
    let modality: String
    let is_gguf: Bool
    let is_gated: Bool
    let suggested_pattern: String
    let likes: Int
    let downloads: Int
    var param_count: Int? = nil
    var param_label: String? = nil
    var est_vram_gb: Double? = nil
    var deploy_status: String? = nil
    var deploy_label_zh: String? = nil
    var deploy_label_en: String? = nil

    var localizedDeployLabel: String {
        let zh = (deploy_label_zh?.isEmpty == false) ? deploy_label_zh! : "显存待探测"
        let en = (deploy_label_en?.isEmpty == false) ? deploy_label_en! : "VRAM TBD"
        return stripEmojiCharacters(L(zh, en))
    }

    var deployBadgeColor: Color {
        switch deploy_status ?? "unknown" {
        case "great", "good":
            return .green
        case "a100":
            return Color(red: 0.85, green: 0.62, blue: 0.05)
        case "heavy":
            return .orange
        case "too_large":
            return .red
        default:
            return .secondary
        }
    }
}

func stripEmojiCharacters(_ text: String) -> String {
    let emojis = ["🟢", "🟡", "🟠", "🔴", "⚪", "📦", "📄", "🔥", "🔒", "🔑", "🌐", "♥", "⬇", "ⓘ"]
    var result = text
    for e in emojis {
        result = result.replacingOccurrences(of: e, with: "")
    }
    return result.trimmingCharacters(in: .whitespacesAndNewlines)
}

func cleanHFReadmeExcerpt(_ rawMd: String) -> String {
    var body = rawMd
    if body.hasPrefix("---"),
       let rEnd = body.range(of: "\n---", range: body.index(body.startIndex, offsetBy: 3)..<body.endIndex) {
        body = String(body[rEnd.upperBound...])
    }
    body = body.replacingOccurrences(of: "<!--[\\s\\S]*?-->", with: "", options: .regularExpression)
    body = body.replacingOccurrences(of: "<a[^>]*>\\s*<img[^>]*>\\s*</a>", with: "", options: [.regularExpression, .caseInsensitive])
    body = body.replacingOccurrences(of: "<img[^>]*>", with: "", options: [.regularExpression, .caseInsensitive])
    body = body.replacingOccurrences(of: "<h[1-4][^>]*>(.*?)</h[1-4]>", with: "\n### $1\n", options: [.regularExpression, .caseInsensitive])
    body = body.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
    body = body.replacingOccurrences(of: "\\[!\\[[^\\]]*\\]\\([^\\)]*\\)\\]\\([^\\)]*\\)", with: "", options: .regularExpression)
    body = body.replacingOccurrences(of: "!\\[[^\\]]*\\]\\([^\\)]*\\)", with: "", options: .regularExpression)
    body = body.replacingOccurrences(of: "&nbsp;", with: " ")
        .replacingOccurrences(of: "&amp;", with: "&")
        .replacingOccurrences(of: "&lt;", with: "<")
        .replacingOccurrences(of: "&gt;", with: ">")
        .replacingOccurrences(of: "&quot;", with: "\"")

    var cleanLines: [String] = []
    var totalChars = 0
    for ln in body.components(separatedBy: .newlines) {
        let s = ln.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.isEmpty || s.hasPrefix("```") || s.hasPrefix("|---") || s.hasPrefix("| ---") { continue }
        cleanLines.append(s)
        totalChars += s.count + 1
        if totalChars >= 1400 { break }
    }
    var joined = cleanLines.joined(separator: "\n")
    joined = joined.replacingOccurrences(of: "\\[([^\\]]+)\\]\\([^\\)\\n]*$", with: "$1", options: .regularExpression)
    return joined
}

func renderHFInlineMarkdown(_ raw: String) -> AttributedString {
    var cleaned = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    cleaned = cleaned.replacingOccurrences(of: "\\[([^\\]]+)\\]\\([^\\)\\n]*$", with: "$1", options: .regularExpression)
    let oddBold = cleaned.components(separatedBy: "**").count - 1
    if oddBold % 2 != 0, let lastRange = cleaned.range(of: "**", options: .backwards) {
        cleaned.removeSubrange(lastRange)
    }
    if let attr = try? AttributedString(
        markdown: cleaned,
        options: AttributedString.MarkdownParsingOptions(
            allowsExtendedAttributes: true,
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible
        )
    ) {
        return attr
    }
    var plain = cleaned
    plain = plain.replacingOccurrences(of: "\\[([^\\]]+)\\]\\(([^\\)]+)\\)", with: "$1", options: .regularExpression)
    plain = plain.replacingOccurrences(of: "**", with: "")
    plain = plain.replacingOccurrences(of: "__", with: "")
    plain = plain.replacingOccurrences(of: "`", with: "")
    return AttributedString(plain)
}

let UNSUPPORTED_HF_PIPELINE_TAGS: Set<String> = [
    "image-to-3d", "text-to-3d", "image-segmentation", "object-detection",
    "depth-estimation", "mask-generation", "keypoint-detection", "robotics",
    "reinforcement-learning", "tabular-classification", "tabular-regression",
    "time-series-forecasting", "audio-classification", "automatic-speech-recognition",
    "text-to-speech", "audio-to-audio", "feature-extraction", "sentence-similarity",
    "token-classification", "text-classification", "zero-shot-image-classification",
    "zero-shot-classification", "fill-mask", "question-answering",
    "table-question-answering", "video-classification"
]

let SUPPORTED_TRENDING_PIPELINE_TAGS: Set<String> = [
    "text-generation", "image-text-to-text", "visual-question-answering",
    "any-to-any", "text-to-image", "image-to-image", "text-to-video", "image-to-video"
]

func analyzeHFModelSizing(_ obj: [String: Any], repoId: String, isGGUF: Bool, fallbackStorageGb: Double = 0.0) -> (
    paramCount: Int,
    paramLabel: String,
    estVramGb: Double,
    deployStatus: String,
    deployZh: String,
    deployEn: String
) {
    let ptag = ((obj["pipeline_tag"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    let st = obj["safetensors"] as? [String: Any]
    let gg = obj["gguf"] as? [String: Any]
    var totalParams: Int64 = 0
    if let n = st?["total"] as? NSNumber {
        totalParams = n.int64Value
    } else if let n = gg?["total"] as? NSNumber {
        totalParams = n.int64Value
    }
    if totalParams <= 0 {
        if let regex = try? NSRegularExpression(pattern: "(?:^|[-_/])(\\d+(?:\\.\\d+)?)\\s*([bBmMtT])(?:[-_/\\b]|$)"),
           let match = regex.firstMatch(in: repoId, range: NSRange(repoId.startIndex..., in: repoId)),
           let r1 = Range(match.range(at: 1), in: repoId),
           let r2 = Range(match.range(at: 2), in: repoId),
           let val = Double(repoId[r1]) {
            let unit = repoId[r2].uppercased()
            if unit == "T" { totalParams = Int64(val * 1_000_000_000_000) }
            else if unit == "B" { totalParams = Int64(val * 1_000_000_000) }
            else if unit == "M" { totalParams = Int64(val * 1_000_000) }
        }
    }

    let paramLabel: String
    if totalParams >= 1_000_000_000_000 {
        paramLabel = String(format: "%.1fT", Double(totalParams) / 1_000_000_000_000.0)
    } else if totalParams >= 1_000_000_000 {
        let b = Double(totalParams) / 1_000_000_000.0
        if b >= 20 {
            paramLabel = String(format: "%.0fB", b)
        } else {
            let s = String(format: "%.1fB", b)
            paramLabel = s.replacingOccurrences(of: ".0B", with: "B")
        }
    } else if totalParams >= 1_000_000 {
        paramLabel = String(format: "%.0fM", Double(totalParams) / 1_000_000.0)
    } else {
        paramLabel = ""
    }

    let paramsB = totalParams > 0 ? (Double(totalParams) / 1_000_000_000.0) : 0.0
    var estVramGb: Double = 0.0
    if isGGUF {
        estVramGb = paramsB > 0 ? ((paramsB * 0.62 + 1.2) * 10).rounded() / 10 : 0.0
    } else if let pmap = st?["parameters"] as? [String: Any], !pmap.isEmpty {
        var bytesTotal: Double = 0.0
        for (dtype, rawCnt) in pmap {
            let cnt = (rawCnt as? NSNumber)?.doubleValue ?? 0.0
            let dt = dtype.uppercased()
            if dt.contains("32") { bytesTotal += cnt * 4.0 }
            else if dt.contains("16") { bytesTotal += cnt * 2.0 }
            else if dt.contains("8") { bytesTotal += cnt * 1.0 }
            else if dt.contains("4") { bytesTotal += cnt * 0.55 }
            else { bytesTotal += cnt * 2.0 }
        }
        estVramGb = (((bytesTotal / 1_000_000_000.0) * 1.12 + 1.2) * 10).rounded() / 10
    } else if paramsB > 0 {
        estVramGb = ((paramsB * 2.1 + 1.2) * 10).rounded() / 10
    } else if fallbackStorageGb > 0 {
        estVramGb = ((fallbackStorageGb * 1.12 + 1.0) * 10).rounded() / 10
    }

    let status: String
    let zh: String
    let en: String
    let vramInt = max(1, Int(estVramGb.rounded()))
    if !isGGUF && UNSUPPORTED_HF_PIPELINE_TAGS.contains(ptag) {
        status = "too_large"
        zh = "非内置管线 (\(ptag) · 需专用SDK)"
        en = "Custom Pipeline (\(ptag))"
    } else if estVramGb <= 0 {
        status = "unknown"
        zh = "显存待探测"
        en = "VRAM TBD"
    } else if estVramGb <= 14.2 {
        status = "great"
        zh = "适合 · T4(15G)可跑 (~\(vramInt)G)"
        en = "Fit: T4 15G (~\(vramInt)G)"
    } else if estVramGb <= 22.5 {
        status = "good"
        zh = "适合 · 需L4(24G) (~\(vramInt)G)"
        en = "Fit: L4 24G (~\(vramInt)G)"
    } else if estVramGb <= 39.0 {
        status = "a100"
        zh = "可部署 · 需A100(40G) (~\(vramInt)G)"
        en = "Fit: A100 40G (~\(vramInt)G)"
    } else if estVramGb <= 78.0 {
        status = "heavy"
        zh = "需A100(80G) (~\(vramInt)G)"
        en = "Needs A100 80G (~\(vramInt)G)"
    } else if isGGUF && paramsB <= 240.0 {
        status = "heavy"
        zh = "需A100(80G)+CPU卸载 (~\(vramInt)G)"
        en = "A100 80G + Offload (~\(vramInt)G)"
    } else {
        status = "too_large"
        zh = "超单卡上限 (~\(vramInt)G · 建议搜GGUF)"
        en = "Too Large (~\(vramInt)G · Try GGUF)"
    }
    return (Int(clamping: totalParams), paramLabel, estVramGb, status, zh, en)
}

struct HFTrendingAPIResponse: Codable {
    var ok: Bool?
    var category: String?
    var models: [HFTrendingModelItem]?
}

struct HFModelWeightFileItem: Identifiable, Codable, Hashable {
    var id: String { filename }
    let filename: String
    let size_gb: Double
    let fit_status: String
    let fit_zh: String
    let fit_en: String

    var localizedFit: String { stripEmojiCharacters(L(fit_zh, fit_en)) }

    var fitColor: Color {
        switch fit_status {
        case "great", "good": return .green
        case "a100": return Color(red: 0.85, green: 0.62, blue: 0.05)
        case "heavy": return .orange
        case "too_large": return .red
        default: return .secondary
        }
    }
}

struct HFModelDetailInfo: Identifiable, Codable, Hashable {
    var id: String { repo_id }
    var ok: Bool?
    var repo_id: String
    var name: String
    var author: String
    var pipeline_tag: String
    var modality: String
    var is_gguf: Bool
    var is_gated: Bool
    var license: String
    var base_model: String
    var architecture: String
    var context_length: Int
    var languages: [String]
    var likes: Int
    var downloads: Int
    var last_modified: String
    var used_storage_gb: Double
    var param_count: Int
    var param_label: String
    var est_vram_gb: Double
    var deploy_status: String
    var deploy_label_zh: String
    var deploy_label_en: String
    var weight_files: [HFModelWeightFileItem]
    var readme_excerpt: String

    var localizedDeployLabel: String {
        stripEmojiCharacters(L(deploy_label_zh.isEmpty ? "显存待探测" : deploy_label_zh, deploy_label_en.isEmpty ? "VRAM TBD" : deploy_label_en))
    }

    var deployBadgeColor: Color {
        switch deploy_status {
        case "great", "good": return .green
        case "a100": return Color(red: 0.85, green: 0.62, blue: 0.05)
        case "heavy": return .orange
        case "too_large": return .red
        default: return .secondary
        }
    }
}

struct ChatTurnItem: Identifiable, Hashable {
    let id = UUID()
    var role: String
    var content: String
    var thinking: String = ""
    var hasImage: Bool = false
}
