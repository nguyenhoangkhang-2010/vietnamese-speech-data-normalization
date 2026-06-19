def avg_duration(samples):
    if not samples:
        return 0.0
    
    return sum(s["duration"] for s in samples) / len(samples)