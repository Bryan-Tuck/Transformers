import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class TweetDataset(TensorDataset):
    """
    Custom dataset to handle tweets and labels data
    """

    def __init__(self, tweets, labels, tokenizer, max_len=None):
        """
        Initialize the dataset

        :param tweets: list of tweets
        :param labels: list of labels
        :param tokenizer: tokenizer object
        :param max_len: maximum length of the input tokens
        """
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len if max_len is not None else self._compute_max_len()
        self.inputs = self.tokenize_texts()

    def __len__(self):
        """
        :returns: the number of tweets in the dataset
        """
        return len(self.tweets)

    def _compute_max_len(self):
        """
        Compute the maximum length of the tweets and cache it.

        :returns: the 95th percentile of the text lengths
        """
        inputs = [self.tokenizer.encode(
            tweet, add_special_tokens=False) for tweet in self.tweets]
        return int(np.percentile([len(i) for i in inputs], 95))

    def tokenize_texts(self):
        """
        Tokenize the tweets and prepare the input

        :returns: input_ids
        """
        inputs = [self.tokenizer.encode(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length'
        ) for tweet in self.tweets]

        return torch.tensor(inputs)

    def __getitem__(self, idx):
        """
        :param idx: index of the tweet
        :return: input_ids, attention_mask, labels
        """
        input_ids = self.inputs[idx]
        attention_mask = torch.ones(input_ids.shape)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, labels


def get_data_loader(tweets, labels, tokenizer, batch_size, max_len=None):
    """
    Returns a DataLoader for training data

    :param tweets: list of tweets
    :param labels: list of labels
    :param tokenizer: tokenizer object
    :param max_len: maximum length of the input tokens
    :return: DataLoader 
"""
    dataset = TweetDataset(tweets, labels, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
