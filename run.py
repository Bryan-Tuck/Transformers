import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# import custom modules
from dataloader import get_data_loader
from trainer import Trainer
from model import TweetClassifier
from utils import get_method_name_from_file, get_classification_report, save_classification_report, create_plots

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
BATCH_SIZE = 32  # 16, 32, 64, 128, 256
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # Use GPU if available
TOKENIZER = AutoTokenizer.from_pretrained(
    'bert-base-uncased')  # Load the tokenizer
MODEL_TYPE = 'bert-base-uncased'  # model type
EPOCHS = 3  # Number of epochs to train the model
LEARNING_RATE = 2e-5  # 2e-5 is the default learning rate for BERT
N_CLASSES = 2  # Number of classes in the dataset
DIRECTORY = 'olid'  # Directory where the csv files are stored

def load_data(file_path):
    """
    Load the dataset from the given file path and split it into train, validation, and test sets.
    
    :param file_path: str, path to the CSV file containing the dataset
    :return: tuple, (train, val, test) DataFrames
    """
    data = pd.read_csv(file_path, usecols=['tweet_clean', 'label'])
    train, val = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    val, test = train_test_split(val, test_size=0.5, random_state=RANDOM_SEED)

    return train, val, test

def get_data_loaders(train, val, test):
    """
    Create data loaders for train, validation, and test sets.

    :param train: DataFrame, the training set
    :param val: DataFrame, the validation set
    :param test: DataFrame, the test set
    :return: tuple, (train_data_loader, val_data_loader, test_data_loader) PyTorch DataLoader objects
    """
    train_tweets = train.tweet_clean.values
    train_labels = train.label.values
    val_tweets = val.tweet_clean.values
    val_labels = val.label.values
    test_tweets = test.tweet_clean.values
    test_labels = test.label.values

    train_data_loader = get_data_loader(train_tweets, train_labels, TOKENIZER, BATCH_SIZE)
    val_data_loader = get_data_loader(val_tweets, val_labels, TOKENIZER, BATCH_SIZE)
    test_data_loader = get_data_loader(test_tweets, test_labels, TOKENIZER, BATCH_SIZE)

    return train_data_loader, val_data_loader, test_data_loader


def init_trainer_and_model(train_data_loader, val_data_loader, test_data_loader):
    """
    Initialize the model, trainer, criterion, and optimizer.

    :param train_data_loader: DataLoader, the training data loader
    :param val_data_loader: DataLoader, the validation data loader
    :param test_data_loader: DataLoader, the test data loader
    :return: tuple, (trainer, model) Trainer and TweetClassifier objects
    """
    model = TweetClassifier(MODEL_TYPE, n_classes=N_CLASSES)
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    trainer = Trainer(model, train_data_loader, val_data_loader,
                      test_data_loader, criterion, optimizer, DEVICE)

    return trainer, model

def main():
    start_time = time.time()

    for file in os.listdir(DIRECTORY):
        if file.endswith('.csv'):
            file_path = os.path.join(DIRECTORY, file)
            method = get_method_name_from_file(file)
            print(f'Running method {method}')

            # Initialize lists to store the scores for each run
            (precision_macro_scores, recall_macro_scores, f1_macro_scores,
             precision_weighted_scores, recall_weighted_scores, f1_weighted_scores,
             accuracy_scores, mcc_scores) = ([] for _ in range(8))

            # Load your data
            train, val, test = load_data(file_path)

            # Get the data loaders
            train_data_loader, val_data_loader, test_data_loader = get_data_loaders(train, val, test)

            # Initialize the trainer and model
            trainer, model = init_trainer_and_model(train_data_loader, val_data_loader, test_data_loader)

            # Train your model
            trainer.train(num_epochs=EPOCHS, use_scheduler=True)

            # Get the training and validation loss and accuracy
            train_loss_values = trainer.losses['train']
            val_loss_values = trainer.losses['val']
            train_acc_values = trainer.accuracies['train']
            val_acc_values = trainer.accuracies['val']

            # Create the plots
            create_plots(method, train_loss_values, val_loss_values,
                         train_acc_values, val_acc_values)

            # Test your model
            y_test, y_pred = trainer.test()

            # Generate evaluation report
            (precision_macro, recall_macro, f1_macro, precision_weighted,
             recall_weighted, f1_weighted, accuracy_score, matthews_correlation_coefficient) = get_classification_report(y_test, y_pred)

            # Store the scores
            precision_macro_scores.append(precision_macro)
            recall_macro_scores.append(recall_macro)
            f1_macro_scores.append(f1_macro)

            precision_weighted_scores.append(precision_weighted)
            recall_weighted_scores.append(recall_weighted)
            f1_weighted_scores.append(f1_weighted)
            accuracy_scores.append(accuracy_score)
            mcc_scores.append(matthews_correlation_coefficient)

            # Save the scores
            save_classification_report(method, precision_macro, recall_macro,
                                       f1_macro, precision_weighted, recall_weighted,
                                       f1_weighted, accuracy_score, matthews_correlation_coefficient)

            # Save the model
            save_path = f'saved_models\method_{method}.pth'
            torch.save(model.state_dict(), save_path)
            torch.cuda.empty_cache()

    end_time = time.time()
    total_time = end_time - start_time
    print("Total running time:", total_time, "seconds")


if __name__ == '__main__':
    main()

