import requests
from bs4 import BeautifulSoup
import os

def get_lyrics_no_api(artist, song_title):
    artist_clean = artist.replace(" ", "").lower()
    title_clean = song_title.replace(" ", "").lower()
    url = f"https://www.azlyrics.com/lyrics/{artist_clean}/{title_clean}.html"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        lyrics_div = soup.find_all('div', {'class': None})[20]
        return lyrics_div.get_text().strip()
    else:
        return None

artist = input("Nhập tên ca sĩ: ")
song = input("Nhập tên bài hát: ")

lyrics = get_lyrics_no_api(artist, song)

if lyrics:
    os.makedirs("data/raw/transcripts", exist_ok=True)
    file_path = f"data/raw/transcripts/{song}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(lyrics)
    print(f"Đã lưu lời bài hát tại {file_path}")
else:
    print("Không tìm thấy lời bài hát. Hãy kiểm tra tên bài hát/ca sĩ.")