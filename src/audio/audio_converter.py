import os
from pydub import AudioSegment


class AudioConverter:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def convert_to_wav(self, input_path: str, output_path: str = None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        file_ext = input_path.split(".")[-1].lower()

        if file_ext not in ["mp3", "mp4", "wav", "m4a"]:
            raise ValueError("Unsupported format")

        audio = AudioSegment.from_file(input_path)

        audio = audio.set_channels(1)

        audio = audio.set_frame_rate(self.target_sr)

        if output_path is None:
            output_path = input_path.replace(f".{file_ext}", ".wav")

        audio.export(output_path, format="wav")

        return output_path


if __name__ == "__main__":
    converter = AudioConverter(target_sr=16000)

    if os.path.exists("input.mp3"):
        input_file = "input.mp3"
    else:
        input_file = "input.mp4"
    output_file = converter.convert_to_wav(input_file)

    print(f"Converted successfully: {output_file}")