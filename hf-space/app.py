"""
FlowTTS Gradio Demo
腾讯云 FlowTTS 语音合成演示 - BYOK (Bring Your Own Key)
"""

import base64
import io
import json
import tempfile
import wave

import gradio as gr
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.trtc.v20190722 import models, trtc_client

# Constants
MODEL = "flow_01_turbo"
ENDPOINT = "trtc.ai.tencentcloudapi.com"
REGION = "ap-beijing"
MAX_TEXT_LENGTH = 2000


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    """Convert PCM to WAV format."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()


def synthesize(
    text: str,
    secret_id: str,
    secret_key: str,
    sdk_app_id: str,
    voice_id: str,
    speed: float,
    volume: float,
    language: str,
    sample_rate: str,
):
    """Synthesize speech from text using Tencent Cloud FlowTTS."""

    # Validation
    if not text or not text.strip():
        raise gr.Error("请输入要合成的文本")

    if len(text) > MAX_TEXT_LENGTH:
        raise gr.Error(f"文本过长：{len(text)} 字符（最多 {MAX_TEXT_LENGTH}）")

    if not secret_id or not secret_key or not sdk_app_id:
        raise gr.Error("请填写完整的腾讯云凭证（SecretId、SecretKey、SdkAppId）")

    try:
        sdk_app_id_int = int(sdk_app_id)
        sample_rate_int = int(sample_rate)
    except ValueError as e:
        raise gr.Error("SdkAppId 和采样率必须是数字") from e

    try:
        # Create client
        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile()
        http_profile.endpoint = ENDPOINT
        http_profile.reqTimeout = 120

        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile

        client = trtc_client.TrtcClient(cred, REGION, client_profile)

        # Build request
        req = models.TextToSpeechSSERequest()
        params = {
            "Model": MODEL,
            "Text": text.strip(),
            "Voice": {
                "VoiceId": voice_id,
                "Speed": speed,
                "Volume": volume,
                "Language": language,
            },
            "AudioFormat": {
                "Format": "pcm",
                "SampleRate": sample_rate_int,
            },
            "SdkAppId": sdk_app_id_int,
        }
        req.from_json_string(json.dumps(params))

        # Call API and collect audio
        audio_chunks = []
        resp = client.TextToSpeechSSE(req)
        for event in resp:
            if isinstance(event, dict) and "data" in event:
                try:
                    data = json.loads(event["data"].strip())
                    if data.get("Type") == "audio" and data.get("Audio"):
                        audio_chunks.append(base64.b64decode(data["Audio"]))
                    if data.get("IsEnd"):
                        break
                except (json.JSONDecodeError, KeyError):
                    continue

        if not audio_chunks:
            raise gr.Error("未收到音频数据，请检查凭证和参数")

        # Convert to WAV and save to temp file
        pcm_data = b"".join(audio_chunks)
        wav_data = pcm_to_wav(pcm_data, sample_rate=sample_rate_int)

        # Save to temp file and return path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_data)
            return f.name

    except gr.Error:
        raise
    except Exception as e:
        error_msg = str(e)
        if "AuthFailure" in error_msg:
            raise gr.Error("认证失败，请检查 SecretId、SecretKey 和 SdkAppId") from e
        elif "InvalidParameter" in error_msg:
            raise gr.Error(f"参数错误：{error_msg}") from e
        elif "RequestLimitExceeded" in error_msg:
            raise gr.Error("请求频率超限，请稍后再试") from e
        else:
            raise gr.Error(f"合成失败：{error_msg}") from e


# Gradio UI
with gr.Blocks(title="FlowTTS 语音合成") as demo:
    gr.Markdown("""
    # FlowTTS 语音合成

    基于腾讯云 FlowTTS 的文字转语音服务。**需要自带腾讯云凭证 (BYOK)**。

    获取凭证：[腾讯云控制台](https://console.cloud.tencent.com/cam/capi) | [开通 TRTC](https://console.cloud.tencent.com/trtc)
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="输入文本",
                placeholder="请输入要合成的文本（最多 2000 字符）...",
                lines=5,
            )

            with gr.Accordion("腾讯云凭证 (必填)", open=True):
                secret_id = gr.Textbox(
                    label="SecretId",
                    placeholder="AKIDxxxxxxxx",
                    type="password",
                )
                secret_key = gr.Textbox(
                    label="SecretKey",
                    placeholder="xxxxxxxx",
                    type="password",
                )
                sdk_app_id = gr.Textbox(
                    label="SdkAppId (例如: 1400000000)",
                    placeholder="1400000000",
                )

            with gr.Accordion("高级设置", open=False):
                voice_id = gr.Textbox(
                    label="音色 ID",
                    value="v-female-R2s4N9qJ",
                )
                with gr.Row():
                    speed = gr.Slider(
                        label="语速",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                    )
                    volume = gr.Slider(
                        label="音量",
                        minimum=0.0,
                        maximum=10.0,
                        value=1.0,
                        step=0.5,
                    )
                with gr.Row():
                    language = gr.Dropdown(
                        label="语言",
                        choices=["zh", "en", "yue", "ja", "ko", "auto"],
                        value="zh",
                    )
                    sample_rate = gr.Dropdown(
                        label="采样率",
                        choices=["16000", "24000"],
                        value="24000",
                    )

            submit_btn = gr.Button("合成语音", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="合成结果")

    gr.Markdown("""
    ---
    **说明：**
    - 本服务仅提供接口封装，不存储任何凭证和数据
    - 语音合成由腾讯云 FlowTTS 完成，费用由腾讯云收取
    - [GitHub](https://github.com/chicogong/flowtts-byok) | [Replicate](https://replicate.com/chicogong/flow-tts)
    """)

    submit_btn.click(
        fn=synthesize,
        inputs=[
            text_input,
            secret_id,
            secret_key,
            sdk_app_id,
            voice_id,
            speed,
            volume,
            language,
            sample_rate,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
