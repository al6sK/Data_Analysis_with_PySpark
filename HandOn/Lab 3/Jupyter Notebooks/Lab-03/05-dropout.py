# Run with: uv run 05-dropout.py
# Demonstrates the impact of Dropout & recurrent_dropout in an LSTM.


import pathlib

import keras
import matplotlib.pyplot as plt
import numpy as np


# Helper to build an LSTM model
def build_lstm(
    name: str,
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
) -> keras.Model:
    """Creates a single‑layer LSTM with optional dropout."""
    inputs = keras.Input(shape=(seq_len,), name=f"{name}_inp")
    x = keras.layers.Embedding(
        input_dim=len(chars) + 1,  # +1 for padding token
        output_dim=64,
        name=f"{name}_emb",
    )(inputs)

    x = keras.layers.LSTM(
        128,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        name=f"{name}_lstm",
    )(x)

    outputs = keras.layers.Dense(
        len(chars) + 1, activation="softmax", name=f"{name}_out"
    )(x)

    model = keras.Model(inputs, outputs, name=name)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


# Prepare a tiny character‑level corpus

EPOCHS = 12
BATCH = 32
VALID_SPLIT = 0.2

text = """
to be, or not to be, that is the question:
whether 'tis nobler in the mind to suffer
the slings and arrows of outrageous fortune...
"""
text = text.lower()
chars = sorted(set(text))
char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = padding
idx2char = {i: c for c, i in char2idx.items()}

seq_len = 40  # length of each training example
step = 1

X, Y = [], []
for i in range(0, len(text) - seq_len, step):
    X.append([char2idx.get(c, 0) for c in text[i : i + seq_len]])
    Y.append(char2idx.get(text[i + seq_len], 0))

X = np.array(X, dtype=np.int32)
Y = keras.utils.to_categorical(Y, num_classes=len(chars) + 1)


# Build the two comparison models

model_no_dropout = build_lstm("no_dropout")
model_with_dropout = build_lstm("with_dropout", dropout=0.3, recurrent_dropout=0.3)


print("\n=== Training model WITHOUT dropout ===")
history_no = model_no_dropout.fit(
    X,
    Y,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=VALID_SPLIT,
)

print("\n=== Training model WITH dropout ===")
history_yes = model_with_dropout.fit(
    X,
    Y,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=VALID_SPLIT,
)


# Plot the loss curves side‑by‑side

plt.figure(figsize=(10, 6))

# Training loss
plt.plot(
    history_no.history["loss"],
    label="No dropout – train",
    color="tab:blue",
    linestyle="--",
)
plt.plot(
    history_yes.history["loss"],
    label="With dropout – train",
    color="tab:orange",
    linestyle="--",
)

# Validation loss
plt.plot(history_no.history["val_loss"], label="No dropout – val", color="tab:blue")
plt.plot(
    history_yes.history["val_loss"], label="With dropout – val", color="tab:orange"
)

plt.title("Effect of Dropout & Recurrent Dropout on LSTM Training")
plt.xlabel("Epoch")
plt.ylabel("Categorical Cross‑Entropy")
plt.legend()
plt.grid(True)

plt.savefig(pathlib.Path(__file__).with_suffix(".png"))
plt.show()


def final_vals(hist):
    return hist.history["loss"][-1], hist.history["val_loss"][-1]


train_no, val_no = final_vals(history_no)
train_yes, val_yes = final_vals(history_yes)

print("\n=== Final epoch summary ===")
print(f"No dropout – train loss: {train_no:.4f}, val loss: {val_no:.4f}")
print(f"With dropout – train loss: {train_yes:.4f}, val loss: {val_yes:.4f}")


# Optional: generate a short piece of text with each model


def generate(model, seed_text="to be", length=100):
    """Greedy generation using the trained model."""
    seq = [char2idx.get(c, 0) for c in seed_text.lower()]
    seq = seq[-seq_len:]  # keep only last seq_len tokens
    generated = seed_text

    for _ in range(length):
        inp = np.array([seq + [0] * (seq_len - len(seq))])
        probs = model.predict(inp, verbose=0)[0]
        next_id = np.argmax(probs)
        next_char = idx2char.get(next_id, "")
        generated += next_char
        seq = seq[1:] + [next_id]

    return generated


print("\n=== Sample generation (no dropout) ===")
print(generate(model_no_dropout, seed_text="to be", length=80))

print("\n=== Sample generation (with dropout) ===")
print(generate(model_with_dropout, seed_text="to be", length=80))
