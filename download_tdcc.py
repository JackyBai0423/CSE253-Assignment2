import os
import json
import subprocess
from pydub import AudioSegment
from yt_dlp import YoutubeDL
import openai

# ==== 配置部分 ====
openai.api_key = "os.environ.get("OPENAI_API_KEY")"  
MODEL = "gpt-4.1-mini"
INPUT_FILE = "tdcc_links.txt"
OUTPUT_DIR = "tdcc_clips"
OUTPUT_JSONL = "train_gpt.jsonl"
CLIP_DURATION_SEC = 30
TARGET_SR = 16000

# ==== 创建输出目录 ====
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 步骤函数 ====

def download_metadata(url):
    with YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title", ""),
        "uploader": info.get("uploader", ""),
        "upload_year": info.get("upload_date", "")[:4]
    }

def download_audio(url, output_prefix):
    command = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", f"{output_prefix}.%(ext)s",
        url
    ]
    subprocess.run(command, check=True)

def extract_clip(input_path, output_path, duration_sec, sample_rate):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(sample_rate).set_channels(1)
    clip = audio[:duration_sec * 1000]
    clip.export(output_path, format="wav")

def generate_prompt_en(title, uploader, year_hint):
    system_prompt = "You are a music tagging assistant. Given a song's title and uploader, describe the musical characteristics of the first 30 seconds in natural English."
    user_prompt = (
        f"Title: {title}\nUploader: {uploader}\nYear: {year_hint}\n\n"
        f"Please describe the musical style, mood, instrumentation, and energy of the first 30 seconds of this track. "
        f"Do not mention artist names or song titles. Focus on the intro's musical characteristics in a way suitable for a music generation model prompt."
    )

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

# ==== 主流程 ====

def main():
    with open(INPUT_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for i, url in enumerate(urls):
            try:
                print(f"[{i}] Processing: {url}")
                base_name = f"tdcc_{i}"
                raw_audio_path = os.path.join(OUTPUT_DIR, base_name + ".wav")
                clip_path = os.path.join(OUTPUT_DIR, base_name + "_clip.wav")

                # 1. 获取元信息
                metadata = download_metadata(url)
                title = metadata["title"]
                uploader = metadata["uploader"]
                year = metadata["upload_year"]

                # 2. 下载 + 切片
                download_audio(url, os.path.join(OUTPUT_DIR, base_name))
                extract_clip(raw_audio_path, clip_path, CLIP_DURATION_SEC, TARGET_SR)
                os.remove(raw_audio_path)

                # 3. 生成英文 prompt
                prompt = generate_prompt_en(title, uploader, year)
                print(f"✓ {base_name}: {prompt[:60]}...")

                # 4. 写入 JSONL
                out_f.write(json.dumps({
                    "audio": clip_path,
                    "text": prompt
                }, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"❌ Failed [{i}] {url}: {e}")

if __name__ == "__main__":
    main()
