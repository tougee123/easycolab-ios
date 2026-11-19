import Foundation

// Run with the companion PendingPKCEState.swift using swiftc.
@main
struct OAuthCallbackChecks {
    static func main() throws {
        for redirect in ["com.example.app:/oauth2redirect", "https://example.org/callback"] {
            let pending = PendingPKCEState(verifier: "test-verifier", redirectURI: redirect, state: "current")
            let restored = try JSONDecoder().decode(PendingPKCEState.self, from: JSONEncoder().encode(pending))
            let good = redirect + "?state=current&code=sample"
            precondition(restored.authorizationCode(from: good, manualEntry: false) == "sample")
            precondition(restored.authorizationCode(from: "4/sample", manualEntry: true) == "4/sample")
            precondition(restored.authorizationCode(from: "4/sample", manualEntry: false) == nil)
            for url in [redirect + "?code=sample", redirect + "?state=old&code=sample", good + "&state=current",
                        good + "&code=other", good + "&error=denied", good + "#fragment",
                        redirect + "/other?state=current&code=sample", "other:/callback?state=current&code=sample"] {
                precondition(restored.authorizationCode(from: url, manualEntry: true) == nil)
            }
        }
        print("OAuth callback checks passed")
    }
}
