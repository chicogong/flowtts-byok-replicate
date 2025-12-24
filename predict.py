"""
FlowTTS BYOK Replicate Wrapper

This is a Cog predictor that wraps Tencent Cloud's FlowTTS (TRTC) API.
Users must provide their own Tencent Cloud credentials (BYOK - Bring Your Own Key).

The wrapper:
1. Accepts user's Tencent Cloud credentials as Secret inputs
2. Calls the FlowTTS SSE streaming API
3. Collects PCM audio chunks from the stream
4. Converts to WAV format and returns the file
"""

import base64
import io
import json
import wave

from cog import BasePredictor, Input, Path, Secret
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.trtc.v20190722 import models, trtc_client

# Constants
MODEL = "flow_01_turbo"
ENDPOINT = "trtc.ai.tencentcloudapi.com"
REGION = "ap-beijing"
MAX_TEXT_LENGTH = 2000  # Character limit per Tencent Cloud API


def pcm_to_wav(
    pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2
) -> bytes:
    """
    Convert raw PCM audio data to WAV format.

    Args:
        pcm_data: Raw PCM audio bytes (16-bit signed, little-endian)
        sample_rate: Sample rate in Hz (16000 or 24000)
        channels: Number of audio channels (1 = mono)
        sample_width: Bytes per sample (2 = 16-bit)

    Returns:
        WAV format audio bytes with proper header
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()


class Predictor(BasePredictor):
    """
    Cog Predictor for FlowTTS text-to-speech synthesis.

    This predictor wraps Tencent Cloud's FlowTTS API, allowing users to
    synthesize speech by providing their own Tencent Cloud credentials.
    """

    def setup(self) -> None:
        """
        Called once when the container starts.
        No model weights to load - this is a thin API wrapper.
        """
        pass

    def predict(
        self,
        # Required inputs
        text: str = Input(description="Text to synthesize (max 2000 characters)"),
        # BYOK credentials (Secret type for security)
        secret_id: Secret = Input(description="Tencent Cloud SecretId (BYOK)"),
        secret_key: Secret = Input(description="Tencent Cloud SecretKey (BYOK)"),
        sdk_app_id: int = Input(description="Tencent Cloud TRTC SdkAppId (BYOK)"),
        # Voice configuration (optional with sensible defaults)
        voice_id: str = Input(
            default="v-female-R2s4N9qJ",
            description="Voice ID (e.g., v-female-R2s4N9qJ)",
        ),
        speed: float = Input(
            default=1.0, ge=0.5, le=2.0, description="Speech speed [0.5, 2.0]"
        ),
        volume: float = Input(
            default=1.0, ge=0.0, le=10.0, description="Volume [0, 10]"
        ),
        pitch: int = Input(
            default=0, ge=-12, le=12, description="Pitch adjustment [-12, 12] semitones"
        ),
        language: str = Input(
            default="zh",
            choices=["zh", "en", "yue", "ja", "ko", "auto"],
            description="Language: zh/en/yue/ja/ko/auto",
        ),
        sample_rate: int = Input(
            default=24000,
            choices=[16000, 24000],
            description="Audio sample rate: 16000 or 24000 Hz",
        ),
        timeout: int = Input(
            default=120,
            ge=10,
            le=300,
            description="Request timeout in seconds [10, 300]",
        ),
    ) -> Path:
        """
        Synthesize speech from text using Tencent Cloud FlowTTS.

        Args:
            text: The text to convert to speech
            secret_id: Tencent Cloud SecretId
            secret_key: Tencent Cloud SecretKey
            sdk_app_id: TRTC application ID
            voice_id: Voice identifier
            speed: Speech speed multiplier
            volume: Audio volume
            pitch: Pitch adjustment in semitones
            language: Target language code
            sample_rate: Output audio sample rate
            timeout: API request timeout

        Returns:
            Path to the generated WAV audio file
        """
        # Input validation
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text too long: {len(text)} characters (max {MAX_TEXT_LENGTH})"
            )

        # Create Tencent Cloud client with user's credentials
        cred = credential.Credential(
            secret_id.get_secret_value(), secret_key.get_secret_value()
        )

        http_profile = HttpProfile()
        http_profile.endpoint = ENDPOINT
        http_profile.reqTimeout = timeout

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
                "Pitch": pitch,
                "Language": language,
            },
            "AudioFormat": {
                "Format": "pcm",
                "SampleRate": sample_rate,
            },
            "SdkAppId": sdk_app_id,
        }
        req.from_json_string(json.dumps(params))

        # Call SSE streaming API and collect audio chunks
        audio_chunks = []
        try:
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
        except Exception as e:
            # Re-raise with cleaner message (avoid exposing credentials)
            error_msg = str(e)
            if "AuthFailure" in error_msg:
                raise RuntimeError(
                    "Authentication failed. Please check your SecretId, "
                    "SecretKey, and SdkAppId."
                ) from e
            elif "InvalidParameter" in error_msg:
                raise RuntimeError(f"Invalid parameter: {error_msg}") from e
            elif "RequestLimitExceeded" in error_msg:
                raise RuntimeError(
                    "Rate limit exceeded. Please try again later."
                ) from e
            else:
                raise RuntimeError(f"TTS API error: {error_msg}") from e

        # Validate we got audio data
        if not audio_chunks:
            raise RuntimeError(
                "No audio data received from upstream API. "
                "Please check your credentials and parameters."
            )

        # Convert PCM to WAV
        pcm_data = b"".join(audio_chunks)
        wav_data = pcm_to_wav(pcm_data, sample_rate=sample_rate)

        # Write to output file
        output_path = Path("/tmp/output.wav")
        with open(output_path, "wb") as f:
            f.write(wav_data)

        return output_path
