import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


# Load GloVe embeddings
def load_glove_embeddings(glove_file):
    embeddings = {}
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype="float32")
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


def build_training_data(data_path: str = "./data/snli_1.0/snli_1.0_train.jsonl"):
    data = pd.read_json(data_path, lines=True)
    # remove all rows where gold_label is '-'
    data = data[data["gold_label"] != "-"]
    sentences1 = data["sentence1"].values
    sentences2 = data["sentence2"].values
    labels = data["gold_label"].values

    # Print rows with missing values or NaN
    missing_rows = data[data.isnull().any(axis=1)]
    if not missing_rows.empty:
        print("Rows with missing values:")
        print(missing_rows)
    else:
        print("No missing values found.")

    # Preprocess the labels
    label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
    labels = np.array([label_map[label] for label in labels])

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        list(zip(sentences1, sentences2)), labels, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val


def train_model():
    # Load GloVe embeddings
    glove_file = "./data/glove.6B//glove.6B.300d.txt"
    embeddings_index = load_glove_embeddings(glove_file)
    embedding_dim = 300

    # Create a tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(embeddings_index.keys())
    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique tokens.")

    # Build the training data
    X_train, X_val, y_train, y_val = build_training_data()
    print("Training data built.")

    premise_embeddings = [
        sentence_embedding(s[0].lower(), embeddings_index) for s in X_train
    ]
    hypothesis_embeddings = [
        sentence_embedding(s[1].lower(), embeddings_index) for s in X_train
    ]
    X_train_embeddings = np.hstack(
        (np.array(premise_embeddings), np.array(hypothesis_embeddings))
    )

    premise_embeddings_val = [
        sentence_embedding(s[0].lower(), embeddings_index) for s in X_val
    ]
    hypothesis_embeddings_val = [
        sentence_embedding(s[1].lower(), embeddings_index) for s in X_val
    ]
    X_val_embeddings = np.hstack(
        (np.array(premise_embeddings_val), np.array(hypothesis_embeddings_val))
    )

    # Convert labels to one-hot encoding
    num_classes = 3
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes)
    print("Labels converted to one-hot encoding.")

    # Define the model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(600,)
            ),  # input shape is twice the GloVe embedding dimension for premise and hypothesis
            tf.keras.layers.Dense(512, activation="relu", input_shape=(embedding_dim,)),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation="relu"),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    print("Model defined.")

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("Model compiled.")

    # Train the model
    history = model.fit(
        X_train_embeddings,
        y_train_one_hot,
        validation_data=(X_val_embeddings, y_val_one_hot),
        epochs=10,
        batch_size=32,
    )
    print("Model trained.")

    # Save the model
    model.save("./data/models/sentence_model.h5")
    print("Model saved.")

    return model, history


def main():
    # Train the model
    model, history = train_model()

    # Plot training & validation accuracy values
    # plt.plot(history.history["accuracy"])
    # plt.plot(history.history["val_accuracy"])
    # plt.title("Model accuracy")
    # plt.ylabel("Accuracy")
    # plt.xlabel("Epoch")
    # plt.legend(["Train", "Validation"], loc="upper left")
    # plt.show()
    # print("Training and validation accuracy plotted.")


if __name__ == "__main__":
    main()
