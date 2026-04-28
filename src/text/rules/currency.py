import re

def normalize_currency(text):
    return re.sub(r"(\d+)k", r"\1 nghìn", text)