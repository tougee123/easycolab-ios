import UIKit

@main
enum StudioPortChecks {
    @MainActor static func main() throws {
        let vm = IOSStudioViewModel()
        let modelIDs = OPEN_MODEL_CATALOG.map(\.id)
        precondition(Set(modelIDs).count == modelIDs.count)
        let h3 = OPEN_MODEL_CATALOG.first { $0.id == "minimax_h3" }!
        precondition(Set(["ref2va", "i2va", "l2va", "fl2va"]).isSubset(of: Set(h3.modes.map(\.id))))
        precondition(modelIDs.contains("qwen38_27b_gguf"))
        vm.switchSelectedModel(OPEN_MODEL_CATALOG.first { $0.id == GPT_IMAGE_MODEL_ID }!)
        precondition(vm.isCloudAPIModel && !vm.isConnected)
        precondition(vm.splitGPTPrompts("first\n---\nsecond") == ["first", "second"])
        let size = h3ImageSize(aspect: "16:9", tier: "4MP")
        precondition(size.width > size.height && size.width % 32 == 0 && size.height % 32 == 0)

        precondition(vm.ref2vaValidationProblem() != nil)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let sample = Data([0, 1, 2])
        try sample.write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let clip = RefMediaItem(url: file, duration: 3, hasAudio: true, sizeMB: 0.001)
        vm.ref2vaVideos = [clip]
        vm.ref2vaAudios = [clip]
        precondition(vm.ref2vaValidationProblem() == nil)
        precondition(vm.ref2vaAudioTags() == ["<Audio 2>"])
        var payload: [String: Any] = [:]
        try vm.addRef2VAPayload(&payload)
        precondition(payload["reference_videos_base64"] as? [String] == [sample.base64EncodedString()])
        vm.ref2vaUseVideoAudio = false
        precondition(vm.ref2vaAudioTags() == ["<Audio 1>"])
        vm.ref2vaVideos = [RefMediaItem(url: file, duration: 16, hasAudio: false, sizeMB: 0.001)]
        precondition(vm.ref2vaValidationProblem() != nil)
        print("iOS studio port checks passed")
    }
}
