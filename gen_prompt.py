import os
import zipfile
import requests
import pandas as pd
import subprocess
import json
from tqdm import tqdm
from openai import OpenAI

# ======== CONFIG ========
FMA_MEDIUM_URL = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma_metadata.zip"
TARGET_GENRES = {"Electronic", "Ambient", "House", "Techno", "Instrumental"}
WAV_DIR = "fma_wav"
JSONL_FILE = "train_gpt.jsonl"
FFMPEG_PATH = r"C:\Users\yello\Downloads\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
MAX_TRACKS = 1000

# ======== OpenAI GPT Setup ========
client = OpenAI()  # Replace with your real key

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded to {dest}")
    else:
        print(f"Already downloaded: {dest}")

def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")
    else:
        print(f"Already extracted: {extract_to}")

def convert_mp3_to_wav(mp3_path, wav_path):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    if not os.path.exists(wav_path):
        cmd = [FFMPEG_PATH, "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def resolve_id_list(s):
    if not isinstance(s, str):
        return []
    try:
        return json.loads(s.replace("'", '"'))
    except:
        return []

def gpt_caption(prompt_info):
    prompt = (
        f"Track title: '{prompt_info['title']}' by {prompt_info['artist']}. "
        f"Album: {prompt_info['album_title']}. "
        f"Tags: {', '.join(prompt_info['tags'])}. "
    )
    if prompt_info['track_info']:
        prompt += f"Track Info: {prompt_info['track_info']} "
    if prompt_info['album_info']:
        prompt += f"Album Info: {prompt_info['album_info']} "

    prompt += (
        "Write a short vivid one-sentence music description (within 80 tokens) that introduces this track to a listener."
        "You should focus on musical element since your description will be used for a music generation model"
        "Mention mood, instrumentation, and atmosphere."
        "You should consider your word as a prompt, that should be used for music generation model to create similar music to the track"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a music critic who writes engaging, short music blurbs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip().replace('\n', ' ')
    except Exception as e:
        print(f"⚠️ GPT error for track '{prompt_info['title']}': {e}")
        return "An electronic track with immersive textures."

def main():
    os.makedirs("downloads", exist_ok=True)
    download_file(FMA_MEDIUM_URL, "downloads/fma_medium.zip")
    download_file(FMA_METADATA_URL, "downloads/fma_metadata.zip")
    extract_zip("downloads/fma_medium.zip", "fma_medium")
    extract_zip("downloads/fma_metadata.zip", "fma_metadata")

    # Load metadata
    meta = pd.read_csv("fma_metadata/tracks.csv", header=[0, 1], index_col=0)
    df = meta['track']
    df = df[df['genre_top'].isin(TARGET_GENRES)]
    genres_df = pd.read_csv("fma_metadata/genres.csv").set_index("genre_id")
    genre_id_to_name = genres_df["title"].to_dict()

    # Prepare data
    os.makedirs(WAV_DIR, exist_ok=True)
    with open(JSONL_FILE, 'w', encoding='utf-8') as fout:
        for i, (track_id, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            if i >= MAX_TRACKS:
                break
            tid = f"{track_id:06d}"
            mp3_path = os.path.join("fma_medium", tid[:3], f"{tid}.mp3")
            wav_path = os.path.join(WAV_DIR, f"{tid}.wav")
            if not os.path.exists(mp3_path):
                continue
            convert_mp3_to_wav(mp3_path, wav_path)

            # Construct metadata
            genre_top = row.get("genre_top", "")
            title = str(row.get("title", "Untitled")).strip()
            artist = str(meta.at[track_id, ("artist", "name")]).strip()
            album_title = str(meta.at[track_id, ("album", "title")]).strip()
            track_info = str(row.get("information", "")).strip()
            album_info = str(meta.at[track_id, ("album", "information")]).strip()

            genres_all = resolve_id_list(str(row.get("genres_all", "[]")))
            tags_id = resolve_id_list(str(row.get("tags", "[]")))
            all_tag_names = set()

            for gid in genres_all + tags_id:
                if gid in genre_id_to_name:
                    all_tag_names.add(genre_id_to_name[gid])
            if genre_top:
                all_tag_names.add(genre_top)

            prompt_info = {
                "title": title,
                "artist": artist or "Unknown",
                "album_title": album_title or "Unknown",
                "tags": list(all_tag_names) or [genre_top],
                "track_info": track_info,
                "album_info": album_info
            }

            caption = gpt_caption(prompt_info)
            fout.write(json.dumps({"audio": wav_path.replace(os.sep, "/"), "text": caption}, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"✅ Finished: {JSONL_FILE} written.")

if __name__ == "__main__":
    convert_mp3_to_wav('fma_wav/001486.wav', os.path.join(WAV_DIR, "001486.wav"))
