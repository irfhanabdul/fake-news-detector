import pandas as pd
import numpy as np
import json



from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import encode_texts
from model import build_model

# 🔹 Load data
df = pd.read_csv("data/real_news_dataset.csv")

# 🔹 Label mapping
df['label'] = df['label'].map({"Fake": 1, "Real": 0})

# 🔹 Remove nulls
df = df.dropna()

texts = df['text'].astype(str).values
labels = df['label'].values

# 🔹 Train-test split (better)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 🔹 Tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# 🔹 Encode
X_train_pad = encode_texts(X_train, tokenizer)
X_test_pad = encode_texts(X_test, tokenizer)

# 🔹 Class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print("Class weights:", class_weight_dict)

# 🔹 Model
model = build_model(10000, 150)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🔹 Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# 🔹 Train
history = model.fit(
    X_train_pad,
    y_train,
    validation_split=0.2,   # 🔥 important
    epochs=6,
    batch_size=16,          # 🔥 optimized for your laptop
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# 🔹 Evaluate
loss, acc = model.evaluate(X_test_pad, y_test)
print("Test Accuracy:", acc)

# 🔹 Save model
model.save("models/fake_news_model.h5")

# 🔹 Save tokenizer
with open("tokenizer/tokenizer.json", "w") as f:
    f.write(tokenizer.to_json())

print("✅ Model & Tokenizer saved successfully!")