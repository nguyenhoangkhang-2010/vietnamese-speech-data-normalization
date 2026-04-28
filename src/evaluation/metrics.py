def avg_duration(samples):
    return sum(s["duration"] for s in samples) / len(samples)