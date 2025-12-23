# -*- coding: utf-8 -*-
"""
FlowTTS Streamlit Demo
腾讯云 FlowTTS 语音合成演示 - BYOK (Bring Your Own Key)
"""

import io
import json
import wave
import base64
import streamlit as st
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.trtc.v20190722 import trtc_client, models

# Constants
MODEL = "flow_01_turbo"
ENDPOINT = "trtc.ai.tencentcloudapi.com"
REGION = "ap-beijing"
MAX_TEXT_LENGTH = 2000

# Page config
st.set_page_config(
    page_title="FlowTTS 语音合成",
    page_icon="🎙️",
    layout="centered",
)


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
    sdk_app_id: int,
    voice_id: str,
    speed: float,
    volume: float,
    language: str,
    sample_rate: int,
) -> bytes:
    """Synthesize speech from text using Tencent Cloud FlowTTS."""
    
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
            "SampleRate": sample_rate,
        },
        "SdkAppId": sdk_app_id,
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
        raise ValueError("未收到音频数据")
    
    # Convert to WAV
    pcm_data = b"".join(audio_chunks)
    return pcm_to_wav(pcm_data, sample_rate=sample_rate)


# UI
st.title("🎙️ FlowTTS 语音合成")
st.markdown("""
基于腾讯云 FlowTTS 的文字转语音服务。**需要自带腾讯云凭证 (BYOK)**。

获取凭证：[腾讯云控制台](https://console.cloud.tencent.com/cam/capi) | [开通 TRTC](https://console.cloud.tencent.com/trtc)
""")

# Input form
with st.form("tts_form"):
    text = st.text_area(
        "输入文本",
        placeholder="请输入要合成的文本（最多 2000 字符）...",
        height=150,
    )
    
    st.subheader("腾讯云凭证")
    col1, col2 = st.columns(2)
    with col1:
        secret_id = st.text_input("SecretId", type="password")
    with col2:
        secret_key = st.text_input("SecretKey", type="password")
    
    sdk_app_id = st.number_input("SdkAppId", min_value=0, step=1, format="%d")
    
    with st.expander("高级设置"):
        voice_id = st.text_input("音色 ID", value="v-female-R2s4N9qJ")
        
        col3, col4 = st.columns(2)
        with col3:
            speed = st.slider("语速", 0.5, 2.0, 1.0, 0.1)
        with col4:
            volume = st.slider("音量", 0.0, 10.0, 1.0, 0.5)
        
        col5, col6 = st.columns(2)
        with col5:
            language = st.selectbox("语言", ["zh", "en", "yue", "ja", "ko", "auto"], index=0)
        with col6:
            sample_rate = st.selectbox("采样率", [16000, 24000], index=1)
    
    submitted = st.form_submit_button("合成语音", type="primary")

# Process
if submitted:
    # Validation
    if not text or not text.strip():
        st.error("请输入要合成的文本")
    elif len(text) > MAX_TEXT_LENGTH:
        st.error(f"文本过长：{len(text)} 字符（最多 {MAX_TEXT_LENGTH}）")
    elif not secret_id or not secret_key or not sdk_app_id:
        st.error("请填写完整的腾讯云凭证")
    else:
        with st.spinner("正在合成语音..."):
            try:
                wav_data = synthesize(
                    text=text,
                    secret_id=secret_id,
                    secret_key=secret_key,
                    sdk_app_id=int(sdk_app_id),
                    voice_id=voice_id,
                    speed=speed,
                    volume=volume,
                    language=language,
                    sample_rate=sample_rate,
                )
                
                st.success("合成成功！")
                st.audio(wav_data, format="audio/wav")
                
                # Download button
                st.download_button(
                    label="下载音频",
                    data=wav_data,
                    file_name="flowtts_output.wav",
                    mime="audio/wav",
                )
                
            except Exception as e:
                error_msg = str(e)
                if "AuthFailure" in error_msg:
                    st.error("认证失败，请检查 SecretId、SecretKey 和 SdkAppId")
                elif "InvalidParameter" in error_msg:
                    st.error(f"参数错误：{error_msg}")
                elif "RequestLimitExceeded" in error_msg:
                    st.error("请求频率超限，请稍后再试")
                else:
                    st.error(f"合成失败：{error_msg}")

# Footer
st.markdown("---")
st.markdown("""
**说明：** 
- 本服务仅提供接口封装，不存储任何凭证和数据
- 语音合成由腾讯云 FlowTTS 完成，费用由腾讯云收取
- [GitHub](https://github.com/chicogong/flowtts-byok) | [Replicate](https://replicate.com/chicogong/flow-tts) | [Hugging Face](https://huggingface.co/spaces/gonghaoran/flow-tts)
""")
