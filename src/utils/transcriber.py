import whisper
import pathlib
import os
from src.utils.separator import separate_vocals

def fetch_lyrics(artist, song, save_path):
    save_path_obj = pathlib.Path(save_path)
    audio_path = save_path_obj.parent.parent / "audio" / f"{song}.mp4"
    
    try:
        vocal_path_str = separate_vocals(str(audio_path))
        vocal_path = pathlib.Path(vocal_path_str)
    except Exception as e:
        print(f"Lỗi tách nhạc: {e}")
        return False
    
    if not vocal_path.exists() or os.path.getsize(vocal_path) < 1000:
        print(f"Lỗi: File âm thanh {vocal_path} không tồn tại hoặc quá nhỏ.")
        return False

    try:
        print(f"--- Đang bắt đầu Transcribe với Whisper ---")
        model = whisper.load_model("large-v3")
        
        result = model.transcribe(
            str(vocal_path), 
            language=None,
            initial_prompt="Đây là bài hát đa ngôn ngữ, bao gồm tiếng Việt và tiếng Anh."
        )
        text = result["text"].strip()
        
        if not text:
            print("Cảnh báo: Whisper không nhận diện được nội dung (kết quả rỗng).")
            return False
            
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"--- Đã lưu lời bài hát tại: {save_path} ---")
        return True
        
    except Exception as e:
        print(f"Lỗi trong quá trình Transcribe: {e}")
        return False