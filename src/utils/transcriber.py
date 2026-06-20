from faster_whisper import WhisperModel
import pathlib
import os
from src.utils.separator import separate_vocals

def fetch_lyrics(artist, song, save_path):
    save_path_obj = pathlib.Path(save_path)
    audio_path = save_path_obj.parent.parent / "audio" / f"{song}.mp4"
    
    try:
        vocal_path_str = separate_vocals(str(audio_path))
        search_dir = pathlib.Path("data/raw/audio/separated/htdemucs")
        found_files = list(search_dir.rglob("vocals.mp3"))
        vocal_path = next((f for f in found_files if song in str(f)), None)
    except Exception as e:
        print(f"Lỗi tách nhạc: {e}")
        return False
    
    if not vocal_path or not vocal_path.exists() or os.path.getsize(vocal_path) < 10000:
        print(f"Lỗi: Không tìm thấy file vocals.mp3 hợp lệ tại {vocal_path}")
        return False

    try:
        print(f"--- Đang bắt đầu Transcribe bằng Faster-Whisper ---")
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        
        segments, info = model.transcribe(
            str(vocal_path), 
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            temperature=0
        )
        
        text = " ".join([seg.text for seg in segments]).strip()
        
        if not text:
            print("Cảnh báo: Không nhận diện được lời (có thể file vocal bị im lặng).")
            return False
            
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"--- Đã lưu lời bài hát tại: {save_path} ---")
        return True
        
    except Exception as e:
        print(f"Lỗi hệ thống trong quá trình Transcribe: {e}")
        return False