import os

def is_valid_audio(path):
    return os.path.exists(path) and path.endswith(".wav")