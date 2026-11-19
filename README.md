# EasyColab iOS

我是 tougee123，负责维护 EasyColab 的 iPhone 和 iPad 版本。这个 SwiftUI 客户端把我在 Mac 工作台上使用的创作功能带到 iOS 15 及以上设备。模型在所连接的 Colab 实例中运行，GPT-Image 单独通过用户自己的 OpenAI API 账户调用。

## 在手机上使用

- 在「云算力」中连接已有实例，或绑定 Google 账号并部署随应用打包的服务端。手动连接需输入实例的 HTTPS 地址和管理密码。Drive 授权上传可单独开启。
- 在「创作」中选择模型。H3 支持首帧、尾帧、首尾帧和多素材参考；参考视频和音频从「文件」选取。ComfyUI 支持安装工作流、查看队列和打开节点画布。
- GPT-Image 可直接生成或编辑图片。输入自己的 API Key 后可刷新账号可用模型。提示词和所选参考图会发给 OpenAI，可能产生 API 费用。
- iOS 挂起应用后会暂停客户端轮询和保活。回到前台可恢复连接并查询云端任务。

## 构建

安装 Xcode 和 iOS 模拟器，在项目根目录运行：

```sh
bash ios_app/run_ios_simulator.sh
```

使用 `bash ios_app/build_ios_ipa.sh` 生成需要自行签名的 IPA。真机安装脚本为 `ios_app/install_release_iphone.sh`。如需原生 Google 登录，请在构建环境中设置自己的 `GOOGLE_OAUTH_CLIENT_ID`，使用为此应用注册的 iOS 客户端 ID 和对应 Bundle ID；该 ID 的回调 scheme 将写入应用。手动连接已有实例无需 Google 登录。

## 登录回调检查

```sh
swiftc ios_app/Sources/PendingPKCEState.swift ios_app/Tests/OAuthCallbackChecks.swift -o /tmp/easycolab-callback-checks
/tmp/easycolab-callback-checks
```

云端入口的鉴权检查可在装有 nginx 和 OpenSSL 的 Linux 环境运行 `python3 -m unittest discover -s tests`。项目使用 [Apache-2.0 许可证](LICENSE)。
