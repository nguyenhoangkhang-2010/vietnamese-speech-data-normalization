import os

def load_dataset(cfg):
    data = []

    for f in os.listdir(cfg["audio_dir"]):
        if f.endswith((".wav", ".mp4")):
            audio_path = os.path.join(cfg["audio_dir"], f)
            base_name = os.path.splitext(f)[0]
            text_path = os.path.join(cfg["text_dir"], f"{base_name}.txt")
            text = ""
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as t:
                    text = t.read().strip()

            data.append({
                "audio_path": audio_path,
                "text": text
            })

    return data