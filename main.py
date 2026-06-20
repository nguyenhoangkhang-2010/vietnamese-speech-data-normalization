import argparse
import pathlib
from pipeline.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Hệ thống xử lý lời bài hát tự động.")
    
    parser.add_argument("--input", type=str, required=True, 
                        help="Tên bài hát (không đuôi) hoặc 'all' để xử lý toàn bộ")
    
    args = parser.parse_args()
    config_path = "configs/config.yaml"
    audio_dir = pathlib.Path("data/raw/audio")

    if args.input == "all":
        song_list = [f.stem for f in audio_dir.glob("*.mp4")]
    else:
        song_list = [args.input]

    for song in song_list:
        print(f"\n>>> Đang xử lý: {song}")
        try:
            run_pipeline(config_path, song)
        except Exception as e:
            print(f"Lỗi khi xử lý {song}: {e}")

if __name__ == "__main__":
    main()