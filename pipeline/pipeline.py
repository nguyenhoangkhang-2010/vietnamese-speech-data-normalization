import yaml
import os
import pathlib
import requests
from bs4 import BeautifulSoup
from src.utils.logger import get_logger

from src.audio.loader import load_dataset
from src.audio.processor import AudioProcessor
from src.text.normalizer import TextNormalizer
from src.alignment.aligner import NemoAligner
from src.dataset.builder import ManifestBuilder
from src.evaluation.report import generate_report

logger = get_logger()

def auto_fetch_lyrics(audio_path):
    path = pathlib.Path(audio_path)
    transcript_path = path.parent.parent / "transcripts" / f"{path.stem}.txt"
    
    if os.path.exists(transcript_path) and os.path.getsize(transcript_path) > 1:
        return str(transcript_path)

    filename = path.stem.lower()
    if '_' in filename:
        parts = filename.split('_')
        artist = parts[0].strip()
        song = parts[1].strip()
    else:
        artist = path.parent.name.lower()
        if artist == "audio": 
            artist = "unknown_artist"
        song = filename.replace("-", "").strip()

    logger.info(f"Đang tìm lời cho bài: {song} của {artist}")
    url = f"https://www.azlyrics.com/lyrics/{artist}/{song}.html"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            if "captcha" in response.text.lower():
                logger.error(f"AZLyrics trả về Captcha cho bài {song}. Bỏ qua.")
            else:
                main_div = soup.find('div', {'class': 'col-xs-12 col-sm-8 col-md-8 col-lg-8'})
                if main_div:
                    lyrics_div = main_div.find_all('div')[5]
                    lyrics = lyrics_div.get_text().strip()
                    
                    os.makedirs(transcript_path.parent, exist_ok=True)
                    with open(transcript_path, "w", encoding="utf-8") as f:
                        f.write(lyrics)
                    return str(transcript_path)
    except Exception as e:
        logger.warning(f"Không thể tải lời tự động: {e}")
    
    os.makedirs(transcript_path.parent, exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(" ")
    return str(transcript_path)

def run_pipeline(config_path):
    logger.info("Loading config...")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_dataset(cfg["dataset"])

    for item in data:
        transcript_path = auto_fetch_lyrics(item["audio_path"])
        with open(transcript_path, "r", encoding="utf-8") as f:
            item["text"] = f.read()

    audio_proc = AudioProcessor(cfg)
    text_norm = TextNormalizer(cfg)
    aligner = NemoAligner(cfg)
    builder = ManifestBuilder(cfg)

    processed = []
    for item in data:
        audio = audio_proc.process(item["audio_path"])
        text = text_norm.normalize(item["text"])
        aligned = aligner.align(audio, text)

        processed.append({
            "audio_filepath": item["audio_path"],
            "text": aligned["text"],
            "duration": aligned["duration"]
        })

    builder.build(processed)
    generate_report(processed)
    logger.info("Pipeline completed")