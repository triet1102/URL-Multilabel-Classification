import torch
from transformers import CamembertModel


class CamemBertMultilabelClassification(torch.nn.Module):
    def __init__(self, nb_class):
        super(CamemBertMultilabelClassification, self).__init__()
        self.nb_class = nb_class
        self.l1 = CamembertModel.from_pretrained("camembert-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, self.nb_class)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
