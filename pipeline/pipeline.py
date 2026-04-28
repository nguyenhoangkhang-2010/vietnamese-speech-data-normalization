import yaml
from src.utils.logger import get_logger

from src.audio.loader import load_dataset
from src.audio.processor import AudioProcessor
from src.text.normalizer import TextNormalizer
from src.alignment.aligner import NemoAligner
from src.dataset.builder import ManifestBuilder
from src.evaluation.report import generate_report

logger = get_logger()


def run_pipeline(config_path):
    logger.info("Loading config...")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_dataset(cfg["dataset"])

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