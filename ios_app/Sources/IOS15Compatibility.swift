import SwiftUI
import PhotosUI

struct StudioNavigation<Content: View>: View {
    @ViewBuilder var content: () -> Content

    var body: some View {
        if #available(iOS 16, *) {
            NavigationStack { content() }
        } else {
            NavigationView { content() }.navigationViewStyle(.stack)
        }
    }
}

extension View {
    @ViewBuilder func studioKeyboardDismissal() -> some View {
        if #available(iOS 16, *) { scrollDismissesKeyboard(.immediately) } else { self }
    }

    @ViewBuilder func studioEditorBackground() -> some View {
        if #available(iOS 16, *) { scrollContentBackground(.hidden) } else { self }
    }

    @ViewBuilder func studioSheetSizes() -> some View {
        if #available(iOS 16, *) { presentationDetents([.medium, .large]) } else { self }
    }
}

struct StudioPromptField: View {
    let title: String
    @Binding var text: String
    let lines: ClosedRange<Int>

    var body: some View {
        if #available(iOS 16, *) {
            TextField(title, text: $text, axis: .vertical).lineLimit(lines)
        } else {
            ZStack(alignment: .topLeading) {
                TextEditor(text: $text).accessibilityLabel(Text(title))
                if text.isEmpty {
                    Text(title).foregroundColor(.secondary).padding(.horizontal, 5)
                        .padding(.vertical, 8).allowsHitTesting(false).accessibilityHidden(true)
                }
            }
            .frame(height: CGFloat(lines.upperBound) * UIFont.preferredFont(forTextStyle: .body).lineHeight + 16)
        }
    }
}

struct StudioPhotoButton<Label: View>: View {
    @Binding var image: UIImage?
    @ViewBuilder var label: () -> Label
    @State private var showingPicker = false

    var body: some View {
        Button { showingPicker = true } label: { label() }
            .sheet(isPresented: $showingPicker) {
                StudioPhotoPicker(image: $image)
            }
    }
}

private struct StudioPhotoPicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) private var dismiss

    init(image: Binding<UIImage?>) { _image = image }

    func makeCoordinator() -> Coordinator { Coordinator(parent: self) }

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ controller: PHPickerViewController, context: Context) {}

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: StudioPhotoPicker
        init(parent: StudioPhotoPicker) { self.parent = parent }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.dismiss()
            guard let provider = results.first?.itemProvider, provider.canLoadObject(ofClass: UIImage.self) else { return }
            provider.loadObject(ofClass: UIImage.self) { object, _ in
                guard let image = object as? UIImage else { return }
                DispatchQueue.main.async { self.parent.image = image }
            }
        }
    }
}
