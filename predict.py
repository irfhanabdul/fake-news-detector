import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from preprocess import encode_texts

# Load model
model = load_model("models/fake_news_model.h5")

# Load tokenizer
with open("tokenizer/tokenizer.json") as f:
    tokenizer = tokenizer_from_json(f.read())

# Predict function
def predict_news(text):
    seq = encode_texts([text], tokenizer)
    prob = model.predict(seq)[0][0]

    percentage = round(float(prob) * 100, 2)
    print("DEBUG → prob:", prob, "| percentage:", percentage)


    if prob > 0.8:
        label = "Fake News ❌"
    elif prob < 0.2:
        label = "Real News ✅"
    else:
        label = "Uncertain 🤔"

    return label, percentage

# Test
while True:
    text = input("Enter news: ")
    if text == "exit":
        break

    label, percentage = predict_news(text)
    print(f"{label} ({percentage}%)")