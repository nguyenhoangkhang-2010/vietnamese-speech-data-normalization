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
from src.utils.lyrics_fetcher import fetch_lyrics

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
    
    try:
        success = fetch_lyrics(artist, song, str(transcript_path))
        
        if success:
            logger.info(f"Đã tải thành công lời bài hát cho: {song}")
        else:
            logger.warning(f"Không tìm thấy lời bài hát cho: {song}. Tạo file rỗng.")
            os.makedirs(transcript_path.parent, exist_ok=True)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(" ")
                
    except Exception as e:
        logger.warning(f"Đã xảy ra lỗi khi tải lời bài hát: {e}")
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