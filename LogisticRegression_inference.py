import os
import pickle
from preprocess import extract_url


def main(save_model_dir):
    with open(os.path.join(save_model_dir, 'cls_tfidf.pkl'), 'rb') as f:
        cls_tfidf, tfidf_vectorizer = pickle.load(f)

    url = input("Enter an URL: ")
    while url:
        url = extract_url(url)
        print(f"Extracted url: {url}")
        url = tfidf_vectorizer.transform([url])
        predicted = cls_tfidf.predict(url)
        print(f"Raw predicted: {predicted}")
        print(f"Index of predicted classes: {[idx for idx, val in enumerate(predicted[0]) if val == 1]}\n")
        url = input("Enter an URL: ")


if __name__ == "__main__":
    DATA_DIR = "data/"
    SAVE_DATA_DIR = "data_preprocessed/"
    SAVE_MODEL_DIR = "saved_models/"
    main(save_model_dir=SAVE_MODEL_DIR)
