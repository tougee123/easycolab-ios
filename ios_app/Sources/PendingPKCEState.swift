import Foundation

struct PendingPKCEState: Codable, Equatable {
    let verifier: String
    let redirectURI: String
    let state: String

    func authorizationCode(from input: String, manualEntry: Bool) -> String? {
        let text = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        guard var callback = URLComponents(string: text), callback.scheme != nil else {
            // Only the manual entry sheet may supply a bare OAuth code.
            let delimiters = CharacterSet(charactersIn: "?:&#=").union(.whitespacesAndNewlines)
            return manualEntry && text.rangeOfCharacter(from: delimiters) == nil ? text : nil
        }
        let items = callback.queryItems ?? []
        callback.query = nil
        let codes = items.filter { $0.name == "code" }
        let states = items.filter { $0.name == "state" }
        guard callback.string == redirectURI, !items.contains(where: { $0.name == "error" }),
              states.count == 1, states[0].value == state,
              codes.count == 1, let code = codes[0].value, !code.isEmpty else { return nil }
        return code
    }
}
