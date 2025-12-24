# FlowTTS BYOK

[![Replicate](https://replicate.com/chicogong/flow-tts/badge)](https://replicate.com/chicogong/flow-tts)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/gonghaoran/flow-tts)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://flowtts.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chicogong/flowtts-byok/blob/master/colab/FlowTTS_Demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

基于腾讯云 FlowTTS 的语音合成服务封装，支持 BYOK（Bring Your Own Key）模式。

## 在线体验

| 平台 | 链接 | 说明 |
|------|------|------|
| **Replicate** | https://replicate.com/chicogong/flow-tts | API + Playground |
| **Hugging Face** | https://huggingface.co/spaces/gonghaoran/flow-tts | 免费 Gradio 演示 |
| **Streamlit** | https://flowtts.streamlit.app/ | Streamlit 演示 |
| **Google Colab** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chicogong/flowtts-byok/blob/master/colab/FlowTTS_Demo.ipynb) | 交互式 Notebook |

## 项目结构

```
flowtts-byok/
├── README.md           # 主文档
├── cog.yaml            # Replicate 配置
├── predict.py          # Replicate 代码
├── hf-space/           # Hugging Face Space 部署代码
│   ├── Dockerfile
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── streamlit/          # Streamlit 版本（可自行部署）
│   ├── app.py
│   └── requirements.txt
├── colab/              # Google Colab Notebook
│   └── FlowTTS_Demo.ipynb
└── examples/           # 示例输出
    └── sample_output.wav
```

## 概述

这是一个部署在多个平台上的 TTS（文字转语音）模型封装。

**重要说明：**
- 本模型**不提供任何 API 密钥**
- 用户必须**自带腾讯云凭证**（BYOK）
- 这是一个 wrapper，实际语音合成由腾讯云 FlowTTS 完成
- 输入的 SSE 流式 PCM 音频会被拼接并转换为 WAV 文件返回
- **不需要 GPU**，运行在 CPU 上

## 前置条件

使用前，你需要准备：

1. **腾讯云账号** - [注册地址](https://cloud.tencent.com/)
2. **开通 TRTC 服务** - [控制台](https://console.cloud.tencent.com/trtc)
3. **获取凭证：**
   - `SecretId` - API 密钥 ID（[获取地址](https://console.cloud.tencent.com/cam/capi)）
   - `SecretKey` - API 密钥
   - `SdkAppId` - TRTC 应用 ID

## 使用方法

### 方式一：Hugging Face Space（推荐体验）

访问 https://huggingface.co/spaces/gonghaoran/flow-tts ，填写凭证即可在线体验。

### 方式二：Google Colab（推荐开发者）

点击下方按钮直接在 Colab 中运行：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chicogong/flowtts-byok/blob/master/colab/FlowTTS_Demo.ipynb)

### 方式三：Replicate Web Playground

访问 https://replicate.com/chicogong/flow-tts ，在表单中填写：
- `text`: 要合成的文本
- `secret_id`: 你的腾讯云 SecretId
- `secret_key`: 你的腾讯云 SecretKey
- `sdk_app_id`: 你的 TRTC SdkAppId

### 方式四：cURL 调用

```bash
curl -s -X POST "https://api.replicate.com/v1/predictions" \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Prefer: wait" \
  -d '{
    "version": "<MODEL_VERSION>",
    "input": {
      "text": "你好，欢迎使用语音合成服务。",
      "secret_id": "YOUR_TENCENT_SECRET_ID",
      "secret_key": "YOUR_TENCENT_SECRET_KEY",
      "sdk_app_id": 1400000000,
      "voice_id": "v-female-R2s4N9qJ",
      "language": "zh"
    }
  }'
```

### 方式五：Python SDK

```python
import replicate

output = replicate.run(
    "chicogong/flow-tts",
    input={
        "text": "你好，欢迎使用语音合成服务。",
        "secret_id": "YOUR_TENCENT_SECRET_ID",
        "secret_key": "YOUR_TENCENT_SECRET_KEY",
        "sdk_app_id": 1400000000,
        "voice_id": "v-female-R2s4N9qJ",
        "language": "zh",
        "speed": 1.0,
        "volume": 1.0,
    }
)

# output 是生成的 WAV 文件 URL
print(output)
```

### 方式六：异步调用 + Webhook（推荐长文本）

```python
import replicate

# 创建异步预测
prediction = replicate.predictions.create(
    version="<MODEL_VERSION>",
    input={
        "text": "这是一段很长的文本...",
        "secret_id": "YOUR_TENCENT_SECRET_ID",
        "secret_key": "YOUR_TENCENT_SECRET_KEY",
        "sdk_app_id": 1400000000,
    },
    webhook="https://your-server.com/webhook",
    webhook_events_filter=["completed"]
)

print(f"Prediction ID: {prediction.id}")
# Replicate 会在完成后 POST 到你的 webhook
```

### 方式七：自行部署 Streamlit

如果你想自己部署 Streamlit 版本：

1. Fork 本仓库
2. 访问 [Streamlit Cloud](https://share.streamlit.io/)
3. 选择 `streamlit/app.py` 作为入口文件
4. 部署即可

## 输入参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 要合成的文本（最多 2000 字符） |
| `secret_id` | Secret | 是 | - | 腾讯云 SecretId |
| `secret_key` | Secret | 是 | - | 腾讯云 SecretKey |
| `sdk_app_id` | int | 是 | - | TRTC SdkAppId |
| `voice_id` | string | 否 | `v-female-R2s4N9qJ` | 音色 ID |
| `speed` | float | 否 | `1.0` | 语速 [0.5, 2.0] |
| `volume` | float | 否 | `1.0` | 音量 [0, 10] |
| `pitch` | int | 否 | `0` | 音调 [-12, 12] 半音 |
| `language` | string | 否 | `zh` | 语言：zh/en/yue/ja/ko/auto |
| `sample_rate` | int | 否 | `24000` | 采样率：16000 或 24000 Hz |
| `timeout` | int | 否 | `120` | 超时时间 [10, 300] 秒 |

## 输出

返回 WAV 格式的音频文件 URL。

## 可用音色

| VoiceId | 描述 |
|---------|------|
| `v-female-R2s4N9qJ` | 女声（默认） |

> 更多音色请参考 [腾讯云 FlowTTS 文档](https://cloud.tencent.com/document/product/647)

## 示例输出

[examples/sample_output.wav](examples/sample_output.wav) - 本地测试生成的示例音频

- 输入文本：`你好，这是一个测试。`
- 格式：WAV (PCM 16-bit, mono, 24000 Hz)
- 时长：1.84 秒

## 本地开发与测试

### 安装 Cog

```bash
# macOS
brew install cog

# Linux
sudo curl -o /usr/local/bin/cog -L \
  "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

### 本地测试

```bash
cog predict \
  -i text="你好，世界" \
  -i secret_id="YOUR_SECRET_ID" \
  -i secret_key="YOUR_SECRET_KEY" \
  -i sdk_app_id=1400000000
```

### 推送到 Replicate

```bash
# 登录
cog login

# 推送
cog push r8.im/chicogong/flow-tts
```

## 安全说明

- `secret_id` 和 `secret_key` 使用 Cog 的 `Secret` 类型，在 Replicate 系统中会被自动脱敏
- 凭证只在容器内使用，不会被记录或返回
- Hugging Face Space 同样不存储任何凭证
- 建议在腾讯云控制台为 API 密钥设置适当的权限和配额限制

## 错误处理

| 错误信息 | 可能原因 |
|----------|----------|
| `Authentication failed` | SecretId/SecretKey/SdkAppId 不正确 |
| `Rate limit exceeded` | 超出腾讯云 API 调用限制 |
| `Text too long` | 文本超过 2000 字符限制 |
| `No audio data received` | 上游 API 未返回音频数据 |

## 技术栈

| 平台 | 技术 |
|------|------|
| **Replicate** | Cog + tencentcloud-sdk-python |
| **Hugging Face** | Docker + Gradio 3.50.2 |
| **Streamlit** | Streamlit + tencentcloud-sdk-python |
| **Colab** | Jupyter Notebook + tencentcloud-sdk-python |

## 许可证

MIT License
