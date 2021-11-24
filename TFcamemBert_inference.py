import tensorflow as tf
from preprocess import extract_url
from transformers import CamembertTokenizer
from TFmodeling import build_camembert_model
import numpy as np


def model_predict(model, ids, mask):
    outputs = model({"input_ids": ids, "attention_mask": mask})
    return outputs


def main(save_model_dir):
    model = build_camembert_model(nb_class=238, seq_length=30, learning_rate=1e-5)
    model.layers[2].load_weights(save_model_dir)
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    url = input("Enter an URL: ")
    while url:
        url = extract_url(url)
        print(f"Extracted url: {url}")

        url_tokenized = tokenizer(url, max_length=30, padding='max_length', truncation=True)
        url_ids = tf.expand_dims(url_tokenized["input_ids"], axis=0)
        url_mask = tf.expand_dims(url_tokenized["attention_mask"], axis=0)

        url_outputs = model_predict(model, url_ids, url_mask)
        url_outputs = np.array(url_outputs) >= 0.5

        print(f"Index of predicted classes: {[idx for idx, val in enumerate(url_outputs[0]) if val == 1]}\n")
        url = input("Enter an URL: ")


if __name__ == "__main__":
    SAVE_MODEL_DIR = "saved_models/tf_camemBert"
    main(save_model_dir=SAVE_MODEL_DIR)
