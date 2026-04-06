from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout

def build_model(vocab_size, max_len):
    model = Sequential([
        Embedding(vocab_size, 256, input_length=max_len),

        # CNN
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(pool_size=2),

        # 🔥 LSTM (increase power)
        LSTM(128, return_sequences=False),

        # 🔥 Dropout (strong)
        Dropout(0.5),

        # Dense
        Dense(64, activation='relu'),

        # 🔥 Extra dropout
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])
    return model