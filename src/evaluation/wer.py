from jiwer import wer

def calculate_wer(ref, hyp):
    return wer(ref, hyp)