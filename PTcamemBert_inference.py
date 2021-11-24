from torch import cuda
from PTmodeling import CamemBertMultilabelClassification
import torch
from transformers import CamembertTokenizer

from preprocess import extract_url

device = 'cuda' if cuda.is_available() else 'cpu'


def main(save_model_path):
    model = CamemBertMultilabelClassification(nb_class=238)
    model.to(device)
    model.load_state_dict(torch.load(save_model_path))
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    url = input("Enter an URL: ")
    while url:
        url = extract_url(url)
        print(f"Extracted url: {url}")
        url_tokenized = tokenizer(url, max_length=30, padding='max_length', truncation=True, return_tensors='pt')
        url_ids = url_tokenized["input_ids"]
        url_mask = url_tokenized["attention_mask"]

        url_outputs = model(url_ids, url_mask)
        print(f"Index of predicted classes: {[idx for idx, val in enumerate(url_outputs[0]) if val == 1]}\n")
        url = input("Enter an URL: ")


if __name__ == "__main__":
    SAVE_MODEL_PATH = "saved_models/pt_camemBert/bert_pt_model.pt"
    main(save_model_path=SAVE_MODEL_PATH)
