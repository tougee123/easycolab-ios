import SwiftUI
import UIKit
import AVFoundation
import UniformTypeIdentifiers

// MARK: - MiniMax-H3 参考生音视频 (ref2va)：参考图 / 参考视频 / 参考音频的选取、校验、请求字段与工作区卡片
// 上限与服务端 app.py 的 H3_REF2VA_LIMITS 一致（官方 README）：图 9 张、视频 3 段、音频 3 段，合计 12 个；
// 视频与音频每段 2~15 秒，同类合计不超过 15 秒。

let H3_REF2VA_MAX_IMAGES = 9
let H3_REF2VA_MAX_VIDEOS = 3
let H3_REF2VA_MAX_AUDIOS = 3
let H3_REF2VA_MAX_TOTAL = 12
let H3_REF2VA_MIN_SECONDS = 2.0
let H3_REF2VA_MAX_SECONDS = 15.0
let H3_REF2VA_MAX_FILE_MB = 100.0   // 视频/音频以 base64 随请求上传，单个文件过大时先压缩

struct RefMediaItem: Identifiable, Hashable {
    let id = UUID()
    let url: URL
    let duration: Double
    let hasAudio: Bool
    let sizeMB: Double
}

enum ReferenceKind { case extra, image, video, audio }

private func fmt(_ n: Double, _ digits: Int) -> String { String(format: "%.*f", digits, n) }

extension IOSStudioViewModel {
    var ref2vaTotalCount: Int { ref2vaImages.count + ref2vaVideos.count + ref2vaAudios.count }

    /// 与服务端/管线一致的编号：图片按顺序 <Picture N>；视频按顺序 <Video N>，带原声的视频依次先占 <Audio N>，独立音频排在后面
    func ref2vaVideoTags() -> [String] {
        var audioNo = 0
        return ref2vaVideos.enumerated().map { idx, item in
            if ref2vaUseVideoAudio && item.hasAudio {
                audioNo += 1
                return "<Video \(idx + 1)> + <Audio \(audioNo)>"
            }
            return "<Video \(idx + 1)>"
        }
    }

    func ref2vaAudioTags() -> [String] {
        let videoAudioCount = ref2vaUseVideoAudio ? ref2vaVideos.filter { $0.hasAudio }.count : 0
        return ref2vaAudios.indices.map { "<Audio \(videoAudioCount + $0 + 1)>" }
    }

    func importReferences(_ urls: [URL], kind: ReferenceKind) async throws {
            for source in urls {
                let scoped = source.startAccessingSecurityScopedResource()
                defer { if scoped { source.stopAccessingSecurityScopedResource() } }
                if kind == .extra || kind == .image {
                    let room = kind == .extra ? GPT_IMAGE_MAX_REFERENCES - 2 - extraReferenceImages.count : min(H3_REF2VA_MAX_IMAGES - ref2vaImages.count, H3_REF2VA_MAX_TOTAL - ref2vaTotalCount)
                    guard room > 0 else { break }
                    guard let image = UIImage(contentsOfFile: source.path) else { throw CocoaError(.fileReadCorruptFile) }
                    if kind == .extra { extraReferenceImages.append(image) } else { ref2vaImages.append(image) }
                    continue
                }
                let video = kind == .video
                let have = video ? ref2vaVideos.count : ref2vaAudios.count
                guard have < (video ? H3_REF2VA_MAX_VIDEOS : H3_REF2VA_MAX_AUDIOS), ref2vaTotalCount < H3_REF2VA_MAX_TOTAL else { break }
                let size = try source.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
                guard size <= 100 * 1048576 else { throw CocoaError(.fileReadTooLarge) }
                let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension(source.pathExtension)
                try FileManager.default.copyItem(at: source, to: url)
                let asset = AVURLAsset(url: url)
                let seconds = (try? await asset.load(.duration)).map { CMTimeGetSeconds($0) } ?? 0
                let audioTracks = (try? await asset.loadTracks(withMediaType: .audio)) ?? []
                let bytes = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                let item = RefMediaItem(
                    url: url,
                    duration: seconds.isFinite ? seconds : 0,
                    hasAudio: !audioTracks.isEmpty,
                    sizeMB: Double(bytes) / 1_048_576.0
                )
                if video { ref2vaVideos.append(item) } else { ref2vaAudios.append(item) }
            }
    }

    /// 返回第一个不符合官方输入规格的问题；全部合规时返回 nil
    func ref2vaValidationProblem() -> String? {
        if ref2vaTotalCount == 0 {
            return L("在「参考生音视频」模式下，请至少添加一张参考图、一段参考视频或一段参考音频。",
                     "In Reference mode, add at least one reference image, video, or audio clip.")
        }
        if ref2vaTotalCount > H3_REF2VA_MAX_TOTAL {
            return L("参考素材合计最多 \(H3_REF2VA_MAX_TOTAL) 个。", "At most \(H3_REF2VA_MAX_TOTAL) references in total.")
        }
        for (kindZh, kindEn, items) in [("视频", "video", ref2vaVideos), ("音频", "audio", ref2vaAudios)] {
            for (i, item) in items.enumerated() {
                if item.duration < H3_REF2VA_MIN_SECONDS - 0.05 || item.duration > H3_REF2VA_MAX_SECONDS + 0.05 {
                    return L("参考\(kindZh) \(i + 1) 时长 \(fmt(item.duration, 1)) 秒，官方要求每段 2~15 秒。",
                             "Reference \(kindEn) \(i + 1) is \(fmt(item.duration, 1)) s; each clip must be 2–15 s.")
                }
                if item.sizeMB > H3_REF2VA_MAX_FILE_MB {
                    return L("参考\(kindZh) \(i + 1) 有 \(fmt(item.sizeMB, 0)) MB，超过 \(Int(H3_REF2VA_MAX_FILE_MB)) MB，请先压缩或截短。",
                             "Reference \(kindEn) \(i + 1) is \(fmt(item.sizeMB, 0)) MB, over \(Int(H3_REF2VA_MAX_FILE_MB)) MB. Compress or trim it first.")
                }
            }
            let total = items.reduce(0) { $0 + $1.duration }
            if total > H3_REF2VA_MAX_SECONDS + 0.05 {
                return L("参考\(kindZh)合计 \(fmt(total, 1)) 秒，官方要求合计不超过 15 秒。",
                         "Reference \(kindEn) clips total \(fmt(total, 1)) s; the limit is 15 s.")
            }
        }
        return nil
    }

    /// 字段名与服务端 GenerateVideoRequest 对应
    func addRef2VAPayload(_ payload: inout [String: Any]) throws {
        payload["reference_images_base64"] = ref2vaImages.compactMap { encodeImageToBase64($0) }
        payload["reference_videos_base64"] = try ref2vaVideos.map { try Data(contentsOf: $0.url).base64EncodedString() }
        payload["reference_audios_base64"] = try ref2vaAudios.map { try Data(contentsOf: $0.url).base64EncodedString() }
        payload["ref2va_use_video_audio"] = ref2vaUseVideoAudio
    }

    func insertRef2VATag(_ tag: String) {
        let sep = visualPrompt.isEmpty || visualPrompt.hasSuffix(" ") || visualPrompt.hasSuffix("\n") ? "" : " "
        visualPrompt += sep + tag + " "
    }
}

