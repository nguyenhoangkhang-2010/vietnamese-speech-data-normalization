import re

def normalize_date(text):
    return re.sub(r"(\d{1,2})/(\d{1,2})", r"ngày \1 tháng \2", text)