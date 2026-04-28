from jiwer import wer

def compute_wer(ref, hyp):
    return wer(ref, hyp)