import lyricsgenius
import os

TOKEN = "oJKvchIXt9bLO7rLD3RwrENSbcHGoceixKT8DN11rCWJeNxPZPsQ4l0VtWoOVOmA"
genius = lyricsgenius.Genius(TOKEN)

def get_lyrics_genius(artist_name, song_name, save_path):
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            return f.read()

    try:
        song = genius.search_song(song_name, artist_name)
        if song:
            lyrics = song.lyrics
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(lyrics)
            return lyrics
    except Exception as e:
        print(f"Lỗi lấy lời từ Genius: {e}")
    return None