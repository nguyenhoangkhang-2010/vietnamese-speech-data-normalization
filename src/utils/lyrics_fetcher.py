import lyricsgenius
import time
import random
import os
from dotenv import load_dotenv


load_dotenv()
GENIUS_TOKEN = os.getenv("GENIUS_TOKEN")

genius = lyricsgenius.Genius(GENIUS_TOKEN, timeout=10)
genius.verbose = False 

def fetch_lyrics(artist, song, save_path):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1:
        return True
    
    wait_time = random.uniform(15,30)
    print(f"--- Đợi {wait_time:.1f}s trước khi tìm lời '{song}' ---")
    time.sleep(wait_time)
    try:
        time.sleep(random.uniform(2, 5))
        song_data = genius.search_song(song, artist)
        if song_data:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(song_data.lyrics)
            return True
    except Exception as e:
        print(f"Lỗi khi gọi API Genius: {e}")
    return False