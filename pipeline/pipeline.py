import yaml
import pathlib
from src.audio.processor import AudioProcessor
from src.text.normalizer import TextNormalizer
from src.alignment.aligner import NemoAligner
from src.dataset.builder import ManifestBuilder
from src.evaluation.report import generate_report
from src.utils.transcriber import fetch_lyrics
from src.utils.logger import get_logger

logger = get_logger()

def run_pipeline(config_path, song_name):
    # 1. Load config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    audio_path = str(pathlib.Path("data/raw/audio") / f"{song_name}.mp4")
    save_path = str(pathlib.Path("data/raw/transcripts") / f"{song_name}.txt")

    # Gọi Transcriber
    success = fetch_lyrics(None, song_name, save_path)
    if not success:
        raise ValueError(f"Pipeline dừng lại: Whisper không thể tạo lời cho {song_name}")

    with open(save_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"File {save_path} trống.")

    # 3. Chạy các module xử lý
    audio_proc = AudioProcessor(cfg)
    text_norm = TextNormalizer(cfg)
    aligner = NemoAligner(cfg)
    builder = ManifestBuilder(cfg)

    # Xử lý
    audio = audio_proc.process(audio_path)
    text = text_norm.normalize(content)
    aligned = aligner.align(audio, text)

    processed = [{
        "audio_filepath": audio_path,
        "text": aligned["text"],
        "duration": aligned["duration"]
    }]

    # 4. Lưu kết quả
    builder.build(processed)
    generate_report(processed)
    logger.info(f"Pipeline hoàn thành cho: {song_name}")