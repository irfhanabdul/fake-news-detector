import pandas as pd

# Load full dataset
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Label add
fake['label'] = "Fake"
real['label'] = "Real"

# IMPORTANT 🔥 (data reduce)
fake_small = fake.sample(8000, random_state=42)
real_small = real.sample(8000, random_state=42)


# Combine
df = pd.concat([fake_small, real_small])

# Keep needed columns
df = df[['text', 'label']]

# Save new dataset
df.to_csv("data/real_news_dataset.csv", index=False)

print("Dataset ready!")