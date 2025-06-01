import os
import json
from yt_dlp import YoutubeDL
import openai

# 设置你的 API key（或从环境变量读取）
from openai import OpenAI

client = OpenAI()
# 视频链接列表（顺序对应 tdcc_0_clip.wav ~ tdcc_31_clip.wav）
video_urls = [
    "https://www.youtube.com/watch?v=Wi3AjJmmsKU",
    "https://www.youtube.com/watch?v=HOsdzEfHyoQ",
    "https://www.youtube.com/watch?v=t0PssM_440M",
    "https://www.youtube.com/watch?v=qQ59qTHqUtc",
    "https://www.youtube.com/watch?v=V94OKMjfAmg",
    "https://www.youtube.com/watch?v=uPg45lN6l6U",
    "https://www.youtube.com/watch?v=L5H1bC6KIyg",
    "https://www.youtube.com/watch?v=5w3jHvYsgBA",
    "https://www.youtube.com/watch?v=xgI_IgZaf6g",
    "https://www.youtube.com/watch?v=_GfRV1WNkEs",
    "https://www.youtube.com/watch?v=UjfVTr7OwsQ",
    "https://www.youtube.com/watch?v=RrJZJtY6u7o",
    "https://www.youtube.com/watch?v=yfQojYPqDNg",
    "https://www.youtube.com/watch?v=lgxg0WH6Pyk",
    "https://www.youtube.com/watch?v=052DB-7GePo",
    "https://www.youtube.com/watch?v=54clWWIf_sY",
    "https://www.youtube.com/watch?v=Dl_C6hPPa2Y",
    "https://www.youtube.com/watch?v=4wH21JH46r4",
    "https://www.youtube.com/watch?v=zKAMxTeSTh4",
    "https://www.youtube.com/watch?v=gVb3Z8vOPiE",
    "https://www.youtube.com/watch?v=T5wLAB9X8y0",
    "https://www.youtube.com/watch?v=jMvW-TIkOJs",
    "https://www.youtube.com/watch?v=R52J5GrPor0",
    "https://www.youtube.com/watch?v=noiy7ajORtA",
    "https://www.youtube.com/watch?v=HW9byxbf-us",
    "https://www.youtube.com/watch?v=kUg72z47lnk",
    "https://www.youtube.com/watch?v=UOuIcLsUtFM",
    "https://www.youtube.com/watch?v=kDvpRmuobwk",
    "https://www.youtube.com/watch?v=VyYgjD_1O7Q",
    "https://www.youtube.com/watch?v=Vn-tK4Lom1I",
    "https://www.youtube.com/watch?v=C1rQC_gKBpI",
    "https://www.youtube.com/watch?v=4FBX3lCxhag"
]

# 输出 JSONL 路径
output_jsonl = "train_gpt.jsonl"
audio_dir = "tdcc_clips"

def fetch_metadata(url):
    cookies_path = os.path.abspath("cookies.txt")  # 确保路径无误
    ydl_opts = {
        'quiet': False,
        'cookiefile': cookies_path
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title", ""),
            "uploader": info.get("uploader", ""),
            "year": info.get("upload_date", "")[:4] if info.get("upload_date") else ""
        }
def generate_prompt(title, uploader, year):
    system_msg = "You are a music tagging assistant. Given a song's info, describe the first 30 seconds as a prompt for music generation."
    user_msg = (
        f"Title: {title}\nUploader: {uploader}\nYear: {year}\n\n"
        "Describe the musical characteristics of the first 30 seconds — style, mood, rhythm, instrumentation, and energy. "
        "Do not mention artist names or song titles. Use natural English, suitable as a prompt for a music generation model."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

def main():
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for idx, url in enumerate(video_urls):
            print(f"Processing {idx}: {url}")
            try:
                metadata = fetch_metadata(url)
                prompt = generate_prompt(metadata["title"], metadata["uploader"], metadata["year"])
                audio_path = os.path.join(audio_dir, f"tdcc_{idx}_clip.wav")
                fout.write(json.dumps({
                    "path": audio_path,
                    "description": prompt
                }, ensure_ascii=False) + "\n")
                print(f"✓ tdcc_{idx}_clip.wav → {prompt[:60]}...")
            except Exception as e:
                print(f"❌ Failed on {url}: {e}")

if __name__ == "__main__":
    main()
