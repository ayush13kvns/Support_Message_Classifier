import re
import unicodedata

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", " emailtoken ", text)
    text = re.sub(r"https?://\S+", " urltoken ", text)
    text = re.sub(r"\b(?:\d[ -]?){10,16}\b", " numbertoken ", text)
    return " ".join(text.split())
