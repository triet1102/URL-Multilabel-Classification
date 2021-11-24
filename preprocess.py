import os
from urllib.parse import urlparse, unquote_plus
import pandas as pd
import numpy as np
import glob
import re
import pickle
import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download("stopwords")
nltk.download("punkt")

STOPWORDS = set(stopwords.words('english') + stopwords.words('french'))
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
REPLACE_IP_ADDRESS = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')


def extract_url(url):
    """
    Extract URL into meaningful words
    :param url: The web URL
    :return: The extracted string
    """
    parsed_unquoted = urlparse(unquote_plus(url))
    text = parsed_unquoted.netloc + ' ' + parsed_unquoted.path + ' ' + parsed_unquoted.params + ' ' + parsed_unquoted.query
    text = text.replace('\n', ' ').lower()
    text = REPLACE_IP_ADDRESS.sub(' ', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = ' '.join([w for w in text.split() if w not in STOPWORDS])
    return text


def convert_tuple(list_label):
    """
    Remove combinations of label that exist only one time
    :param df: Dataframe
    :return: Filtered dataframe
    """
    return tuple(list_label)


def group_less_occur_label(target, counter, threshold=200):
    """
    Group labels that occur less than threshold times
    :param target: Target labels of an URL
    :param counter: Label counter
    :param threshold: Threshold
    :return: New target
    """
    original_len = len(target)
    new_target = list(label for label in target if counter[label] > threshold)
    if len(new_target) < original_len:
        new_target.append('Other')

    return new_target


def preprocess_data(data_dir, save_data_dir):
    """
    Read and preprocess data
    :param data_dir: Path of data directory
    :param save_data_dir: Path to save preprocessed data
    :return: None
    """
    # Ignore _SUCCESS file, read only ["url", "target"] columns
    data = []
    data_files = glob.glob(data_dir + "*.snappy.parquet")
    print(f"Data files: {data_files}")

    for data_file in data_files:
        df = pd.read_parquet(data_file, columns=["url", "target"])
        data.append(df)
    data = pd.concat(data, ignore_index=True)

    # Extract words from URL and save to CSV file
    data["url"] = data["url"].apply(extract_url)

    # Delete duplicated row and row that has combinations of label that exist only one time
    data = data[~data["url"].duplicated()]
    data['target'] = data['target'].apply(convert_tuple)
    data = data.groupby("target").filter(lambda x: len(x) > 1)

    # Group labels that occur less than threshold times and replace them by label 0
    counter = Counter(list(label for target in data["target"].values for label in target))
    data["target"] = data["target"].apply(lambda x: group_less_occur_label(x, counter))

    with open(os.path.join(save_data_dir, 'preprocessed.pkl'), 'wb') as f:
        pickle.dump(data, f)
