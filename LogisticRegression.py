import nltk
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from preprocess import preprocess_data
from util import print_evaluation_scores, get_train_test_val

nltk.download("stopwords")
nltk.download("punkt")


def tfidf_features(X_train, X_val, X_test):
    """
    Create tfidf representation and vocabulary
    :param X_train: X_train examples
    :param X_val: X_val examples
    :param X_test: X_test examples
    :return: tfidf representation and vocabulary for each set of examples
    """
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=10)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer


def train_classifier(X_train, y_train, C, regularisation):
    """
    Train classification model
    :param X_train: X_train examples
    :param y_train: y_train examples
    :param C: Inverse of regularization strength
    :param regularisation: Type of the penalty
    :return: Trained model
    """
    model = OneVsRestClassifier(LogisticRegression(penalty=regularisation, C=C, max_iter=1_000)).fit(X_train, y_train)
    return model


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

    # Split data into train, validation and test
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_val(data)

    # Create tf-idf representation
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_features(X_train, X_val, X_test)

    # Train model
    cls_tfidf = train_classifier(X_train_tfidf, y_train, C=50, regularisation='l2')

    # Save trained model
    with open(os.path.join(save_model_dir, 'cls_tfidf.pkl'), 'wb') as f:
        pickle.dump([cls_tfidf, tfidf_vectorizer], f)

    # Predict and Evaluate train, test set
    y_val_predicted_targets_tfidf = cls_tfidf.predict(X_val_tfidf)
    y_test_predicted_targets_tfidf = cls_tfidf.predict(X_test_tfidf)

    print('TF-IDF')
    # Evaluate model on validation and test set
    print("Evaluation on validation set:")
    print_evaluation_scores(y_val, y_val_predicted_targets_tfidf)
    print("\nEvaluation on test set:")
    print_evaluation_scores(y_test, y_test_predicted_targets_tfidf)

    # Print out some examples
    for i in range(20):
        print(X_test[i])
        y_pred = [idx for idx, val in enumerate(y_test_predicted_targets_tfidf[i]) if val == 1]
        y_true = [idx for idx, val in enumerate(y_test[i]) if val == 1]
        print(y_pred)
        print(y_true)
        print('\n')


if __name__ == "__main__":
    DATA_DIR = "data/"
    SAVE_DATA_DIR = "data_preprocessed/"
    SAVE_MODEL_DIR = "saved_models/"
    main(data_dir=DATA_DIR,
         save_data_dir=SAVE_DATA_DIR,
         save_model_dir=SAVE_MODEL_DIR)
