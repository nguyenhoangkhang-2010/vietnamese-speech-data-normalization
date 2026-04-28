import re

class TextNormalizer:
    def __init__(self, cfg):
        self.cfg = cfg

    def normalize(self, text):
        text = text.lower()

        text = self.normalize_numbers(text)
        text = self.normalize_dates(text)
        text = self.clean(text)

        return text

    def normalize_numbers(self, text):
        text = re.sub(r"\b1\b", "một", text)
        text = re.sub(r"\b2\b", "hai", text)
        text = re.sub(r"\b3\b", "ba", text)
        return text

    def normalize_dates(self, text):
        return re.sub(r"(\d{1,2})/(\d{1,2})", r"ngày \1 tháng \2", text)

    def clean(self, text):
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()