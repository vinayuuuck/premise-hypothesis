import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datasets import load_dataset


# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = "".join(values[:-300])
            vector = np.asarray(values[-300:], dtype="float32")
            embeddings[word] = vector
    return embeddings


def create_embedding_matrix(word_index, embeddings, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def sentence_embedding(sentence, embeddings_index):
    words = sentence.split()
    embedding_dim = next(iter(embeddings_index.values())).shape[0]
    sentence_embedding = np.zeros(embedding_dim)
    for word in words:
        embedding_vec = embeddings_index.get(word)
        if embedding_vec is not None:
            sentence_embedding += embedding_vec
    return (sentence_embedding + 1) / (len(words) + 1)


def sentence_embedding_pooling(sentence, embeddings_index, embedding_dim):
    words = sentence.split()
    vectors = [
        embeddings_index.get(w.lower()) for w in words if w.lower() in embeddings_index
    ]
    if not vectors:
        print(f"Warning: No embeddings found for words in sentence: {sentence}")
        fallback = np.full(
            (embedding_dim,), fill_value=1.0 / (embedding_dim + 1), dtype="float32"
        )
        return fallback, fallback
    stacked = np.stack(vectors, axis=0)
    avg_vec = np.mean(stacked, axis=0)
    max_vec = np.max(stacked, axis=0)

    return avg_vec, max_vec


def combine(premise, hypothesis, embeddings_index, embedding_dim):
    premise_avg, premise_max = sentence_embedding_pooling(
        premise, embeddings_index, embedding_dim
    )
    hypothesis_avg, hypothesis_max = sentence_embedding_pooling(
        hypothesis, embeddings_index, embedding_dim
    )
    abs_diff = np.abs(premise_avg - hypothesis_avg)
    prod = premise_avg * hypothesis_avg

    return np.concatenate(
        [premise_avg, premise_max, hypothesis_avg, hypothesis_max, abs_diff, prod]
    )


def train_model():
    # Load GloVe embeddings
    glove_file = "./data/glove.840B.300d.txt"
    embeddings_index = load_glove_embeddings(glove_file)
    embedding_dim = 300
    print(f"Loaded {len(embeddings_index)} word vectors.")

    # Create a tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(embeddings_index.keys())
    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique tokens.")

    # Build the training data
    snli = load_dataset("snli")
    snli = snli.filter(lambda x: x["label"] != "-1")
    X_train = snli["train"]
    X_val = snli["validation"]
    y_train = X_train["label"]
    y_val = X_val["label"]

    X_train_embeddings = np.array(
        [
            combine(s["premise"], s["hypothesis"], embeddings_index, embedding_dim)
            for s in X_train
        ]
    )

    X_val_embeddings = np.array(
        [
            combine(s["premise"], s["hypothesis"], embeddings_index, embedding_dim)
            for s in X_val
        ]
    )

    # Convert labels to one-hot encoding
    num_classes = 3
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes)
    print("Labels converted to one-hot encoding.")

    # Define the model
    inputs = tf.keras.layers.Input(shape=(X_train_embeddings.shape[1],))
    x = tf.keras.layers.Dense(
        512,
        activation="relu",
    )(inputs)
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
    )(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(
        X_train_embeddings,
        y_train_one_hot,
        validation_data=(X_val_embeddings, y_val_one_hot),
        epochs=10,
        batch_size=32,
    )
    print("Model trained.")

    # Evaluate the model
    test_embeddings = np.array(
        [
            combine(s["premise"], s["hypothesis"], embeddings_index, embedding_dim)
            for s in snli["test"]
        ]
    )
    test_labels = snli["test"]["label"]
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes)

    test_loss, test_accuracy = model.evaluate(test_embeddings, test_labels_one_hot)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

    # Save the model
    model.save("./data/models/sentence_model.h5")
    print("Model saved.")

    return model, history


def main():
    model, history = train_model()


if __name__ == "__main__":
    main()
