import gradio as gr
import soundfile as sf
import torch
import os
import numpy as np
import random
import warnings
import time
import gc # Garbage collection for memory management

# --- 1. SILENCE WARNINGS ---
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

# Global variables to track state
lux_tts = None
active_device = None

# --- HELPER: MODEL LOADER ---
def load_model(target_device):
    global lux_tts, active_device
    
    # Check if we actually need to reload
    if lux_tts is not None and active_device == target_device:
        return lux_tts

    # Safety check for CUDA
    if target_device == "cuda" and not torch.cuda.is_available():
        raise gr.Error("❌ 本系统不支持 CUDA (GPU)，请选择 CPU。")

    print(f"\n🔄 Loading LuxTTS Model on [{target_device.upper()}]...")
    
    # CLEANUP: Free up memory from the old model before loading the new one
    if lux_tts is not None:
        del lux_tts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        from zipvoice.luxvoice import LuxTTS
        # Initialize the model on the requested device
        lux_tts = LuxTTS(device=target_device)  # 优先使用 ckpt/LuxTTS，不存在则自动下载
        active_device = target_device
        print(f"✅ Model successfully loaded on {target_device}")
        return lux_tts

    except Exception as e:
        print(f"Initialization Error: {e}")
        raise gr.Error(f"在 {target_device} 上加载模型失败，请查看终端获取详情。")

# --- HELPER: SEEDING ---
def set_all_seeds(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

# --- GENERATION FUNCTION ---
def clone_voice(text, ref_audio_path, rms, t_shift, num_steps, speed, return_smooth, ref_duration, seed, device_choice):
    
    if not ref_audio_path:
        raise gr.Error("请上传参考音频文件。")

    # 1. Load/Switch Model (Only happens if device changed)
    model = load_model(device_choice)
    
    # 2. Handle Seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    actual_seed = set_all_seeds(seed)
    
    print(f"\n--- Generating (Device: {device_choice} | Seed: {actual_seed}) ---")

    # 3. Encode Prompt
    # (Note: Moving the prompt to the correct device is handled internally by the library usually, 
    # but we ensure the model is on the right device first)
    encoded_prompt = model.encode_prompt(
        ref_audio_path, 
        duration=ref_duration, 
        rms=rms
    )

    # 4. Generate & Time
    start_time = time.time()
    
    final_wav = model.generate_speech(
        text, 
        encoded_prompt, 
        num_steps=int(num_steps), 
        t_shift=t_shift, 
        speed=speed, 
        return_smooth=return_smooth
    )
    
    end_time = time.time()
    generation_time = end_time - start_time

    # 5. Process Output
    audio_data = final_wav.detach().cpu().numpy().squeeze()
    
    # Clip and Convert to 16-bit PCM
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_data = (audio_data * 32767).astype(np.int16)

    # 6. Calculate Stats
    audio_duration_sec = len(audio_data) / 48000
    speedup = audio_duration_sec / generation_time if generation_time > 0 else 0
    
    stats_msg = (
        f"💻 设备:            {device_choice.upper()}\n"
        f"⏱️ 生成耗时:        {generation_time:.4f} 秒\n"
        f"🔊 音频时长:        {audio_duration_sec:.2f} 秒\n"
        f"🚀 实时倍率:        {speedup:.1f}x"
    )
    
    print(stats_msg)
    
    return (48000, audio_data), actual_seed, stats_msg

# --- LONGER EXAMPLE TEXT ---
long_text_example = (
    "在古老森林的深处，阳光透过树冠洒下缕缕金光，照亮了在静止空气中飞舞的尘埃。"
    "唯一的声音是附近小溪的潺潺流水，蜿蜒流过长满青苔、历经数百年不变的石头。"
    "这是一个时间仿佛放慢脚步的地方，让疲惫的旅人忘却外界世界的重担。"
)

# --- STARTUP: DETECT DEFAULT DEVICE ---
default_device = "cuda" if torch.cuda.is_available() else "cpu"

# --- GRADIO INTERFACE ---
with gr.Blocks(title="LuxTTS 语音克隆") as demo:
    gr.Markdown("# 🎙️ LuxTTS 语音克隆")
    gr.Markdown("测试速度、音质与硬件性能（CPU 与 GPU 对比）。")
    
    with gr.Row():
        with gr.Column():
            # Device Selection (Top for visibility)
            device_radio = gr.Radio(
                choices=["cuda", "cpu"], 
                value=default_device, 
                label="计算设备（切换需几秒钟）"
            )

            text_input = gr.Textbox(
                label="待合成文本", 
                value=long_text_example,
                lines=5
            )
            ref_audio_input = gr.Audio(
                label="参考音频", 
                type="filepath"
            )
            
            with gr.Accordion("高级参数 / Advanced Parameters", open=True):
                seed_input = gr.Number(
                    value=-1, precision=0,
                    label="随机种子 / Seed（-1 为随机）",
                    info="控制生成随机性，相同种子可复现相同结果；-1 表示每次随机"
                )
                rms_slider = gr.Slider(
                    0.001, 0.1, value=0.01, step=0.001,
                    label="RMS / 响度归一化",
                    info="参考音频响度目标值，越大输出越响，推荐 0.01"
                )
                speed_slider = gr.Slider(
                    0.1, 3.0, value=0.8, step=0.1,
                    label="语速 / Speed",
                    info="1.0 为正常语速，>1 加速，<1 减速"
                )
                steps_slider = gr.Slider(
                    1, 20, value=4, step=1,
                    label="步数 / Num Steps",
                    info="扩散采样步数，越大音质越好但越慢，3–4 为效率最佳"
                )
                t_shift_slider = gr.Slider(
                    0.1, 2.0, value=0.9, step=0.1,
                    label="T Shift / 时间偏移",
                    info="采样时间偏移，较高可提升音质但可能增加识别错误"
                )
                duration_number = gr.Number(
                    value=5,
                    label="参考时长 / Ref Duration（秒）",
                    info="参考音频截取时长，越小推理越快；出现伪影时可设为 1000"
                )
                smooth_check = gr.Checkbox(
                    label="平滑输出 / Return Smooth",
                    value=False,
                    info="输出更平滑，可减轻金属感，但可能略欠清晰"
                )

            generate_btn = gr.Button("生成语音", variant="primary")

        with gr.Column():
            output_audio = gr.Audio(label="生成结果")
            stats_output = gr.Textbox(label="性能指标", lines=5, interactive=False)
            seed_output = gr.Number(label="实际使用的种子", interactive=False)

    # Initialize model on startup (optional, makes first generation faster)
    # logic: we just print a message, the actual load happens on first click or we can force it here.
    # To keep startup fast, we let the first click handle the load.

    generate_btn.click(
        fn=clone_voice,
        inputs=[
            text_input, ref_audio_input, rms_slider, t_shift_slider, 
            steps_slider, speed_slider, smooth_check, duration_number,
            seed_input, device_radio
        ],
        outputs=[output_audio, seed_output, stats_output]
    )

if __name__ == "__main__":
    demo.launch()