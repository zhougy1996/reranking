from transformers import BertModel, BertConfig
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, bert_path, num_labels):
        super(TextClassifier, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]
        return self.fc(out_pool)
