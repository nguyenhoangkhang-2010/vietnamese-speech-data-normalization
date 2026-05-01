import argparse
from src.text.normalizer import TextNormalizer

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str)
args = parser.parse_args()

n = TextNormalizer({})
print(n.normalize(args.text))