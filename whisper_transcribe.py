#!/usr/bin/env python3
"""
Whisper 视频转录脚本
功能：音频提取 → Whisper英文转录 → 生成LRC

流程：
    Step 1: 转录 (生成英语LRC)

用法:
    python whisper_transcribe.py

依赖：
    - openai-whisper
    - ffmpeg
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] 使用设备: {DEVICE}")


def install_whisper():
    """安装 whisper 及依赖"""
    print("[INFO] 安装 openai-whisper...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "openai-whisper"
    ])
    print("[INFO] whisper 安装完成")


def extract_audio(video_path: str, audio_path: str) -> str:
    """使用 ffmpeg 从视频提取音频（去掉字幕轨道）"""
    print(f"[INFO] 从视频提取音频: {video_path}")

    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-map", "0:a:0",  # 只用第一个音轨
           "-sn",            # 去掉字幕轨道
           "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 音频提取失败: {result.stderr}")

    print(f"[INFO] 音频已提取: {audio_path}")
    return audio_path


def transcribe_audio(audio_path: str, model_name: str = "base", language: str = "en") -> dict:
    """使用 Whisper 将音频转录为文本"""
    import whisper

    print(f"[INFO] 加载 Whisper 模型: {model_name}")
    model = whisper.load_model(model_name, device=DEVICE)

    print(f"[INFO] 开始转录: {audio_path}")
    result = model.transcribe(audio_path, language=language, verbose=False)

    return result


def format_timestamp(seconds: float) -> str:
    """将秒数格式化为 [mm:ss.xx] 格式"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02d}:{secs:05.2f}]"


def generate_lrc_from_segments(segments: list) -> str:
    """从转录片段生成LRC格式字幕（仅英文）"""
    lines = []

    for seg in segments:
        start = seg["start"]
        text = seg["text"].strip()

        if not text:
            continue

        timestamp = format_timestamp(start)
        lines.append(f"{timestamp}{text}")

    return "\n".join(lines)


def main():
    # ========== 硬编码参数 ==========
    VIDEO_PATH = "/home/wlia0047/ar57/wenyu/Shameless.US.S01E02.2011.1080p.Blu-ray.x265.10bit.AC3￡cXcY@FRDS.mkv"
    MODEL_NAME = "large"
    OUTPUT_DIR = Path("/home/wlia0047/ar57/wenyu/transcripts")
    # =================================

    # 根据视频文件名生成输出路径
    video_path_obj = Path(VIDEO_PATH)
    video_name = video_path_obj.name  # 完整视频文件名（含扩展名）
    audio_path = video_path_obj.with_suffix(".wav")  # 音频与视频同目录
    en_lrc_path = OUTPUT_DIR / f"en_{video_name}.lrc"  # 英文LRC
    # =================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 检查英文LRC是否已存在
    if en_lrc_path.exists():
        print(f"[INFO] 英文LRC已存在: {en_lrc_path}")
        print(f"[INFO] 跳过转录")
        return

    # ========== 清理旧文件（只清理transcripts目录，不动源视频） ==========
    for fp in [audio_path]:
        if fp.exists():
            fp.unlink()
            print(f"[CLEAN] 已删除: {fp}")
    # =====================================================================

    # 确保 whisper 已安装
    try:
        import whisper
    except ImportError:
        install_whisper()

    # 提取音频
    extract_audio(str(video_path_obj), str(audio_path))

    # 英文转录
    print(f"[INFO] 英文转录中...")
    result_en = transcribe_audio(str(audio_path), model_name=MODEL_NAME, language="en")
    segments_en = result_en.get("segments", [])

    # 保存英文LRC
    en_lrc_path.write_text(generate_lrc_from_segments(segments_en), encoding="utf-8")

    # 保存JSON
    output_json = en_lrc_path.with_suffix(".json")
    full_result = {
        "english_segments": segments_en,
        "lrc_path": str(en_lrc_path)
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(full_result, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] 英文转录完成!")
    print(f"  - 英文LRC: {en_lrc_path}")
    print(f"  - 条数: {len(segments_en)}")


if __name__ == "__main__":
    main()
