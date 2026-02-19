"""
LuxTTS Legado API 服务

供 Legado 等阅读软件通过 httpTTS 配置调用的 TTS 接口。
启动: LUXTTS_REF_AUDIO=ref_audio/京京.wav uvicorn api_server:app --host 0.0.0.0 --port 8765
注意: 使用 workers=1 避免多进程重复加载模型。
"""

import gc
import io
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response

# 抑制 torch 相关警告
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

# 项目根目录（api_server.py 所在目录的父目录）
PROJECT_ROOT = Path(__file__).resolve().parent

# 全局模型状态
lux_tts = None
active_device = None

# 默认参考音频（环境变量 LUXTTS_REF_AUDIO 或启动时设置）
DEFAULT_REF_AUDIO: str | None = os.environ.get("LUXTTS_REF_AUDIO")

app = FastAPI(
    title="LuxTTS API",
    description="Legado 兼容的 TTS 接口，基于 LuxTTS 语音克隆",
    version="0.1.0",
)


def _load_model(target_device: str):
    """加载 LuxTTS 模型"""
    global lux_tts, active_device

    if lux_tts is not None and active_device == target_device:
        return lux_tts

    if target_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("本系统不支持 CUDA (GPU)，请使用 CPU。")

    print(f"\n🔄 Loading LuxTTS Model on [{target_device.upper()}]...")

    if lux_tts is not None:
        del lux_tts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        from zipvoice.luxvoice import LuxTTS

        lux_tts = LuxTTS(device=target_device)
        active_device = target_device
        globals()["lux_tts"] = lux_tts
        print(f"✅ Model successfully loaded on {target_device}")
        return lux_tts
    except Exception as e:
        print(f"Initialization Error: {e}")
        raise ValueError(f"在 {target_device} 上加载模型失败: {e}") from e


def _resolve_ref_audio_path(path: str) -> Path:
    """解析并校验参考音频路径，防止路径遍历"""
    if ".." in path:
        raise ValueError("路径不允许包含 ..")

    if Path(path).is_absolute():
        resolved = Path(path).resolve()
    else:
        resolved = (PROJECT_ROOT / path).resolve()
        if not str(resolved).startswith(str(PROJECT_ROOT)):
            raise ValueError(f"参考音频路径超出项目范围: {path}")

    if not resolved.exists():
        raise ValueError(f"参考音频文件不存在: {path}")
    if not resolved.is_file():
        raise ValueError(f"路径不是文件: {path}")
    return resolved


def _generate_tts(text: str, ref_audio_path: str, speed: float = 1.0) -> bytes:
    """生成 TTS 音频并返回 WAV 字节"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(device)

    ref_path = _resolve_ref_audio_path(ref_audio_path)

    encoded_prompt = model.encode_prompt(
        str(ref_path),
        duration=5,
        rms=0.01,
    )

    final_wav = model.generate_speech(
        text,
        encoded_prompt,
        num_steps=4,
        t_shift=0.9,
        speed=speed,
        return_smooth=False,
    )

    audio_data = final_wav.detach().cpu().numpy().squeeze()
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_data = (audio_data * 32767).astype(np.int16)

    # 写入 WAV 到内存
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(48000)
        wav_file.writeframes(audio_data.tobytes())

    return buffer.getvalue()


def _get_ref_audio(ref_audio: str | None) -> str:
    """获取有效的参考音频路径"""
    path = ref_audio or DEFAULT_REF_AUDIO
    if not path:
        raise HTTPException(
            status_code=400,
            detail="未配置参考音频。请设置环境变量 LUXTTS_REF_AUDIO 或在请求中传入 ref_audio 参数。",
        )
    return path


@app.get("/api/tts")
async def tts_get(
    text: str = Query(..., description="待合成文本"),
    speed: float = Query(1.0, ge=0.1, le=3.0, description="语速"),
    ref_audio: str | None = Query(None, description="参考音频路径"),
):
    """GET 方式 TTS 接口，Legado 兼容"""
    ref_path = _get_ref_audio(ref_audio)
    try:
        wav_bytes = _generate_tts(text, ref_path, speed)
        return Response(content=wav_bytes, media_type="audio/wav")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/tts")
async def tts_post(request: Request):
    """POST 方式 TTS 接口，支持 JSON 和 form-urlencoded，Legado 兼容"""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        text = body.get("text")
        speed = float(body.get("speed", 1.0))
        ref_audio = body.get("ref_audio")
    else:
        # form-urlencoded
        form = await request.form()
        text = form.get("text")
        speed_val = form.get("speed", "1.0")
        speed = float(speed_val) if speed_val else 1.0
        ref_audio = form.get("ref_audio")

    if not text:
        raise HTTPException(status_code=400, detail="缺少 text 参数")

    ref_path = _get_ref_audio(ref_audio)
    try:
        wav_bytes = _generate_tts(text, ref_path, speed)
        return Response(content=wav_bytes, media_type="audio/wav")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
async def root():
    """健康检查"""
    return {"status": "ok", "service": "LuxTTS API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8765,
        workers=1,
    )
