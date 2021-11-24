import torch
import pandas as pd
import os
import numpy as np
import dill as pickle
from transformers import CamembertTokenizer

from preprocess import preprocess_data
from TFmodeling import build_camembert_model
from util import print_evaluation_scores, get_train_test_val, data_information

MAX_LEN = 30
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-05


def main(data_dir, save_data_dir, save_model_dir):
    """
    Train and Evaluate tfidf approach
    :param data_dir: Path of data directory
    :param save_data_dir: Path to save preprocessed data
    :param save_model_dir: Path to save trained model
    :return: None
    """

    # Preprocess data
    preprocessed_file = os.path.join(save_data_dir, "preprocessed.pkl")
    if not os.path.exists(preprocessed_file):
        preprocess_data(data_dir=data_dir, save_data_dir=save_data_dir)

    # Read data
    with open(preprocessed_file, 'rb') as f:
        data = pickle.load(f)

    print(tuple(data['target'][0]))
    # Show data information
    data_information(data)

    # Split data into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_val(data)

    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    X_train_tokenized = tokenizer(list(X_train), max_length=MAX_LEN, padding='max_length', truncation=True)
    X_val_tokenized = tokenizer(list(X_val), max_length=MAX_LEN, padding='max_length', truncation=True)
    X_test_tokenized = tokenizer(list(X_test), max_length=MAX_LEN, padding='max_length', truncation=True)

    nb_class = len(y_train[0])
    print(nb_class)
    model = build_camembert_model(nb_class=nb_class, seq_length=MAX_LEN, learning_rate=LEARNING_RATE)

    print('OK Setup model')
    history = model.fit({"input_ids": np.array(X_train_tokenized["input_ids"]),
                         "attention_mask": np.array(X_train_tokenized["attention_mask"])},
                        y=np.array(y_train),
                        batch_size=TRAIN_BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=1)

    # Save trained model
    model.layers[2].save_weights(save_model_dir)


if __name__ == "__main__":
    DATA_DIR = "data/"
    SAVE_DATA_DIR = "data_preprocessed/"
    SAVE_MODEL_DIR = "saved_models/tf_camemBert"
    main(data_dir=DATA_DIR,
         save_data_dir=SAVE_DATA_DIR,
         save_model_dir=SAVE_MODEL_DIR)
