import SwiftUI
import UniformTypeIdentifiers

struct StudioModelControls: View {
    @ObservedObject var vm: IOSStudioViewModel
    @State private var picking = false
    @State private var kind = ReferenceKind.image
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if vm.isCloudAPIModel {
                Text("OpenAI API").font(.headline)
                SecureField("OpenAI API Key", text: $vm.openAIKeyDraft)
                    .textInputAutocapitalization(.never).disableAutocorrection(true)
                HStack {
                    Button(L("保存密钥", "Save key")) { vm.saveOpenAIKey() }
                    Button(L("清除", "Remove")) { vm.deleteOpenAIKey() }
                    Button(L("刷新模型", "Refresh models")) { Task { await vm.refreshOpenAIImageModels() } }
                }
                options("Model", $vm.openAIImageModel, vm.openAIImageModels)
                options("Aspect", $vm.gptImageAspect, ["auto"] + GPT_IMAGE_ASPECTS.map(\.id))
                options("Size", $vm.gptImageTier, GPT_IMAGE_TIERS.map(\.id))
                options("Quality", $vm.openAIImageQuality, GPT_IMAGE_QUALITIES.map(\.id))
                options("Complexity", $vm.openAIImageComplexity, GPT_IMAGE_COMPLEXITIES.map(\.id))
            }
            if vm.selectedModelId == "minimax_h3_image" {
                options("Aspect", $vm.h3ImageAspect, H3_IMAGE_ASPECTS.map(\.id))
                options("Size", $vm.h3ImageTier, H3_IMAGE_TIERS.map(\.id))
            }
            if vm.taskModeId == "i2i_edit" && (vm.isCloudAPIModel || vm.selectedModelId == "minimax_h3_image") {
                pickerButton(L("添加更多参考图", "Add reference images"), .extra)
                ForEach(vm.extraReferenceImages.indices, id: \.self) { i in
                    Button("− Picture \(i + 3)") { vm.extraReferenceImages.remove(at: i) }
                }
            }
            if vm.taskModeId == "ref2va" {
                HStack {
                    pickerButton(L("参考图", "Pictures"), .image)
                    pickerButton(L("视频", "Video"), .video)
                    pickerButton(L("音频", "Audio"), .audio)
                }
                Toggle(L("使用视频原声", "Use video audio"), isOn: $vm.ref2vaUseVideoAudio)
                ForEach(vm.ref2vaImages.indices, id: \.self) { i in
                    refRow("<Picture \(i + 1)>") { vm.ref2vaImages.remove(at: i) }
                }
                ForEach(Array(vm.ref2vaVideos.enumerated()), id: \.element.id) { i, item in
                    refRow(vm.ref2vaVideoTags()[i]) { vm.ref2vaVideos.remove(at: i); try? FileManager.default.removeItem(at: item.url) }
                }
                ForEach(Array(vm.ref2vaAudios.enumerated()), id: \.element.id) { i, item in
                    refRow(vm.ref2vaAudioTags()[i]) { vm.ref2vaAudios.remove(at: i); try? FileManager.default.removeItem(at: item.url) }
                }
            }
        }
        .disabled(vm.isGenerating)
        .fileImporter(isPresented: $picking, allowedContentTypes: kind == .video ? [.movie] : kind == .audio ? [.audio] : [.image], allowsMultipleSelection: true) { result in
            Task { do { try await vm.importReferences(try result.get(), kind: kind) }
                catch { vm.statusMessage = error.localizedDescription } }
        }
    }
    private func options(_ label: String, _ value: Binding<String>, _ choices: [String]) -> some View {
        HStack { Text(label); Spacer(); Picker(label, selection: value) {
            ForEach(choices, id: \.self) { Text($0).tag($0) }
        }.pickerStyle(.menu) }
    }
    private func pickerButton(_ label: String, _ type: ReferenceKind) -> some View {
        Button(label) { kind = type; picking = true }
    }
    private func refRow(_ label: String, remove: @escaping () -> Void) -> some View {
        HStack { Button(label) { vm.insertRef2VATag(label) }; Spacer(); Button(role: .destructive, action: remove) { Image(systemName: "trash") } }
    }
}
