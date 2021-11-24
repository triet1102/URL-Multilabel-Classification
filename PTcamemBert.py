import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
import dill as pickle
from tqdm import tqdm
from transformers import CamembertTokenizer
from torch import cuda

from preprocess import preprocess_data
from PTdataset import CustomDataset
from PTmodeling import CamemBertMultilabelClassification
from util import print_evaluation_scores, get_train_test_val

device = 'cuda' if cuda.is_available() else 'cpu'
MAX_LEN = 30
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def evaluate(model, data_loader):
    model.eval()
    data_targets = []
    data_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            data_targets.extend(targets.cpu().detach().numpy().tolist())
            data_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    data_outputs = np.array(data_outputs) >= 0.5
    print_evaluation_scores(data_targets, data_outputs)


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

    X_train = X_train[:20]
    X_val = X_val[:20]
    X_test = X_test[:20]
    y_train = y_train[:20]
    y_val = y_val[:20]
    y_test = y_test[:20]

    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    training_set = CustomDataset(X_train, y_train, tokenizer, MAX_LEN)
    validating_set = CustomDataset(X_val, y_val, tokenizer, MAX_LEN)
    testing_set = CustomDataset(X_test, y_test, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    valid_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': TEST_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    validating_loader = DataLoader(validating_set, **valid_params)
    testing_loader = DataLoader(testing_set, **test_params)

    nb_class = len(y_train[0])
    model = CamemBertMultilabelClassification(nb_class=nb_class)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # Train model
    model.train()
    for i in range(EPOCHS):
        for _, data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if _ % 100 == 0:
                print(f'Epoch: {i}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('save model')
    torch.save(model.state_dict(), save_model_dir)

    # Evaluate model on validation set
    print('\nCamemBert Pytorch\n')
    print("Evaluation on validation set:")
    evaluate(model, validating_loader)
    print("Evaluation on test set:")
    evaluate(model, testing_loader)


if __name__ == "__main__":
    DATA_DIR = "data/"
    SAVE_DATA_DIR = "data_preprocessed/"
    SAVE_MODEL_DIR = "saved_models/pt_camemBert/bert_pt_model.pt"
    main(data_dir=DATA_DIR,
         save_data_dir=SAVE_DATA_DIR,
         save_model_dir=SAVE_MODEL_DIR)
