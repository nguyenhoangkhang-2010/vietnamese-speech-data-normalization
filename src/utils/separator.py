import subprocess
import pathlib
import os

def separate_vocals(audio_path):
    audio_path = pathlib.Path(audio_path)
    output_dir = audio_path.parent / "separated"
    
    cmd = ["demucs", "--two-stems=vocals", "--mp3", str(audio_path), "-o", str(output_dir)]
    subprocess.run(cmd, check=True)
    
    temp_vocal_path = output_dir / "htdemucs" / audio_path.stem / "vocals.mp3"
    final_vocal_path = audio_path.parent / f"{audio_path.stem}_clean.mp3"
    
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(temp_vocal_path),
        "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(final_vocal_path)
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    
    return str(final_vocal_path)