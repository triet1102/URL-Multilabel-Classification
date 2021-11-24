from collections import Counter
import itertools
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, hamming_loss


def data_information(data):
    """
    Show information about input data
    :param data: The dataframe
    :return: None
    """
    # Number of element in data
    print(f'Length of data:  {len(data)}')

    # Shape of data
    print(f'Shape of data: {data.shape}')

    targets = data["target"].values
    target_counts = Counter(list(label for target in targets for label in target))

    # Number of distinct targets
    print(f"Number of distinct targets: {len(target_counts.keys())}")

    # Max target per URL
    max_target_per_url = max(len(i) for i in data["target"])
    print(f"Max target per URL: {max_target_per_url}")

    # Number of URL that have at least one target
    is_target_not_empty = len([target for target in targets if len(target) > 0])
    print(f"Number of URL that have at least one target: {is_target_not_empty}")

    # Top target
    target_counts_sorted = dict(sorted(target_counts.items(), key=lambda item: item[1], reverse=True))
    print(f"Target count sorted: {target_counts_sorted}")
    popular_targets = sorted(target_counts, key=target_counts.get, reverse=True)
    print(f"Top 20 popular targets: {popular_targets[:20]}")

    # Get target weights
    target_weight = {}
    for label in target_counts.keys():
        target_weight[label] = round(target_counts[label] / data.shape[0] * 100, 2)
    print(f"Weight per target: {target_weight}")

    return


def print_evaluation_scores(y_test, predicted):
    print(f"Accuracy: {accuracy_score(y_test, predicted, normalize=True)}")  # Test with normalize=False
    print(f"Hamming loss: {hamming_loss(y_test, predicted)}")
    print(f"F1 score macro: {f1_score(y_test, predicted, average='macro', zero_division=0)}")
    print(f"F1 score micro: {f1_score(y_test, predicted, average='micro', zero_division=0)}")
    print(f"F1 score weighted: {f1_score(y_test, predicted, average='weighted', zero_division=0)}")
    # print(f"Precision macro: {average_precision_score(y_test, predicted, average='macro')}")
    # print(f"Precision micro: {average_precision_score(y_test, predicted, average='micro')}")
    # print(f"Precision weighted: {average_precision_score(y_test, predicted, average='weighted')}")


def get_train_test_val(data):
    """
    Split data into train, validation and test sets
    :param data: Data
    :return: Tuple of [X_train, X_val, X_test, y_train, y_val, y_test]
    """
    X_data = data["url"].values
    y_data = data["target"]
    X, X_test, y, y_test = train_test_split(X_data, y_data, test_size=0.1, train_size=0.9, stratify=y_data,
                                            random_state=200)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, train_size=0.75, random_state=300)

    # Create multi-hot form of labels
    targets = set(itertools.chain(*[i for i in data["target"]]))
    mlb = MultiLabelBinarizer(classes=list(targets))
    y_train = mlb.fit_transform(y_train)
    y_val = mlb.fit_transform(y_val)
    y_test = mlb.fit_transform(y_test)

    print(f"X_train_shape: {X_train.shape}"
          f"\ny_train shape: {y_train.shape}"
          f"\nX_test shape: {X_test.shape}"
          f"\ny_test shape: {y_test.shape}"
          f"\nX_val shape: {X_val.shape}"
          f"\ny_val shape: {y_val.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test
