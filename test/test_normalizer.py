from src.text.normalizer import TextNormalizer

def test_norm():
    n = TextNormalizer({})
    assert "một" in n.normalize("1")