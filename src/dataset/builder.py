import json
import os

class ManifestBuilder:
    def __init__(self, cfg):
        self.path = cfg["output"]["manifest_path"]

    def build(self, data):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")