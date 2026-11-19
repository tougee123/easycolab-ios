# EasyColab for iOS

This directory contains the SwiftUI client from [EasyColab](https://github.com/wy51ai/easycolab). It connects to the user's Colab runtime. This version adds checks to the Google sign-in callback used by the iOS client.

## Build and run

On a Mac with Xcode and an iOS Simulator runtime installed, run these commands from the repository root.

```sh
bash ios_app/run_ios_simulator.sh --build-only
```

To install and launch the app, boot an iOS 15 or newer iPhone simulator in Xcode and run the same script without `--build-only`.

## Login callback checks

The app stores the expected redirect URI, state, and PKCE verifier together for each pending login. A callback URL must match that redirect URI and supply exactly one matching state and one nonempty code. The manual entry sheet also accepts a bare authorization code. A login started on an older build must be restarted after upgrading.

Run the checks with synthetic data, without a Google account:

```sh
swiftc ios_app/Sources/PendingPKCEState.swift ios_app/Tests/OAuthCallbackChecks.swift -o /tmp/easycolab-callback-checks
/tmp/easycolab-callback-checks
```
