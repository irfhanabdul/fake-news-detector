import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = 150

def clean_text(text):
    text = str(text).lower()

    # Remove Reuters
    text = re.sub(r"\(reuters\)", " ", text)

    # Remove location tags
    text = re.sub(r"^[A-Z\s\-]+", " ", text)

    # 🔥 Remove breaking bias
    text = re.sub(r"\bbreaking\b", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def encode_texts(texts, tokenizer):
    texts = [clean_text(t) for t in texts]
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')