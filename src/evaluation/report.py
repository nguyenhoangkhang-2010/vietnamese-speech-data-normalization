import json

from src.evaluation.metrics import avg_duration

def generate_report(samples):
    report = {
        "num_samples": len(samples),
        "avg_duration": avg_duration(samples)
    }

    with open("data/processed/report.json", "w") as f:
        json.dump(report, f, indent=2)