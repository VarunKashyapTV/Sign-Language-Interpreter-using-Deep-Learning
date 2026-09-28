import numpy as np
import os
from glob import glob
from sklearn.utils import shuffle
import pickle


def load_all_sequences():
    files = sorted(glob("gestures_mp/*.npy"))
    if not files:
        raise FileNotFoundError("No gesture .npy files found in gestures_mp/")

    all_sequences = []
    all_labels = []
    for f in files:
        g_id = int(os.path.splitext(os.path.basename(f))[0])
        seqs = np.load(f)  # shape (N, SEQ_LEN, 63)
        all_sequences.append(seqs)
        all_labels.extend([g_id] * len(seqs))
        print(f"Loaded {len(seqs)} sequences for gesture {g_id}")

    sequences = np.concatenate(all_sequences, axis=0)
    labels = np.array(all_labels)
    return sequences, labels


sequences, labels = load_all_sequences()
sequences, labels = shuffle(sequences, labels)
print("Total sequences:", len(sequences))

n = len(sequences)
train_end = int(5 / 6 * n)
test_end = int(11 / 12 * n)

train_sequences, train_labels = sequences[:train_end], labels[:train_end]
test_sequences, test_labels = sequences[train_end:test_end], labels[train_end:test_end]
val_sequences, val_labels = sequences[test_end:], labels[test_end:]

print("Train:", len(train_sequences), "Test:", len(test_sequences), "Val:", len(val_sequences))

with open("train_sequences_mp", "wb") as f:
    pickle.dump(train_sequences, f)
with open("train_labels_mp", "wb") as f:
    pickle.dump(train_labels, f)
with open("test_sequences_mp", "wb") as f:
    pickle.dump(test_sequences, f)
with open("test_labels_mp", "wb") as f:
    pickle.dump(test_labels, f)
with open("val_sequences_mp", "wb") as f:
    pickle.dump(val_sequences, f)
with open("val_labels_mp", "wb") as f:
    pickle.dump(val_labels, f)