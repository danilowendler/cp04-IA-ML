import re
from html import unescape
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_review(text: str) -> str:
    text = unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_vectorizer(max_features, ngram_range, min_df, max_df, sublinear_tf):
    return TfidfVectorizer(
        preprocessor=clean_review,
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )
