import Foundation
import Security

enum RuntimeAccess {
    private final class RedirectPolicy: NSObject, URLSessionTaskDelegate {
        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
            completionHandler(nil)
        }
    }
    static let session = URLSession(configuration: .ephemeral, delegate: RedirectPolicy(), delegateQueue: nil)

    private static func query(_ url: URL?) -> [String: Any]? {
        guard let url, url.scheme?.lowercased() == "https", let host = url.host?.lowercased(), url.user == nil, url.password == nil else { return nil }
        return [kSecClass as String: kSecClassGenericPassword,
                kSecAttrService as String: "easycolab.ios.runtime",
                kSecAttrAccount as String: "\(host):\(url.port ?? 443)"]
    }

    static func password(for url: URL?) -> String? {
        guard var q = query(url) else { return nil }
        q[kSecReturnData as String] = true
        var item: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &item) == errSecSuccess, let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func save(_ value: String, for url: URL?) throws {
        guard var q = query(url) else { throw URLError(.badURL) }
        let fields: [String: Any] = [kSecValueData as String: Data(value.utf8),
                                    kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly]
        var status = SecItemUpdate(q as CFDictionary, fields as CFDictionary)
        if status == errSecItemNotFound {
            q.merge(fields) { _, new in new }
            status = SecItemAdd(q as CFDictionary, nil)
        }
        guard status == errSecSuccess else { throw NSError(domain: NSOSStatusErrorDomain, code: Int(status)) }
    }

    static func authorized(_ req: URLRequest, password: String) -> URLRequest {
        var request = req
        request.setValue("Basic " + Data("easycolab:\(password)".utf8).base64EncodedString(), forHTTPHeaderField: "Authorization")
        return request
    }

    static func request(_ req: URLRequest) -> URLRequest {
        guard req.value(forHTTPHeaderField: "Authorization") == nil,
              let key = password(for: req.url) else { return req }
        return authorized(req, password: key)
    }

    static func data(for req: URLRequest) async throws -> (Data, URLResponse) {
        guard req.value(forHTTPHeaderField: "Authorization") == nil, let password = password(for: req.url) else { return try await URLSession.shared.data(for: req) }
        return try await session.data(for: authorized(req, password: password))
    }
}
