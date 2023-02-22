from torch import nn
from transformers import AutoModel

class TweetClassifier(nn.Module):
    """
    :param model_type: The type of model to be used (e.g., "bert-base-uncased")
    :param n_classes: Number of classes in the dataset
    """

    def __init__(self, model_type, n_classes):
        # Call the parent class constructor
        super(TweetClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: A batch of input ids
        :param attention_mask: A batch of attention masks
        """
        # Feed the input to the model
        _, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.dropout(pooled_output)
        return self.fc(output)

class PerspectiveClassifier(nn.Module):
    pass