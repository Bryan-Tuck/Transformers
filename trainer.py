import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_data_loader, val_data_loader, test_data_loader,
                 criterion, optimizer, device):
        """
        Initialize the trainer class.

        :param model: the model to be trained
        :param train_data_loader: DataLoader for training data
        :param val_data_loader: DataLoader for validation data
        :param test_data_loader: DataLoader for test data
        :param criterion: Loss function to be used
        :param optimizer: Optimization algorithm to be used
        :param device: device to be used (cpu/gpu)
        """
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()  # mixed precision training
        self.losses = {'train': [], 'val': []}  # to store losses
        self.accuracies = {'train': [], 'val': []}  # to store accuracies
        # number of batches in training set
        self.num_train_batches = len(self.train_data_loader)
        # number of batches in validation set
        self.num_val_batches = len(self.val_data_loader)

    def train(self, num_epochs, use_scheduler=True):
        """ 
        :param num_epochs: number of epochs to train the model
        :param use_scheduler: whether to use learning rate scheduler or not
        """
        self.model.train()  # set model to training mode
        # total number of training steps
        num_training_steps = self.num_train_batches * num_epochs
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=num_training_steps)  # learning rate scheduler
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            train_loss = 0.0
            train_acc = 0.0
            for _, (inputs, masks, labels) in tqdm(enumerate(self.train_data_loader),
                                                   total=self.num_train_batches,
                                                   desc=f'Epoch {epoch+1}/{num_epochs}'):  # loop over batches
                inputs = inputs.to(self.device)  # move inputs to device
                masks = masks.to(self.device)  # move masks to device
                labels = labels.to(self.device)  # move labels to device

                self.optimizer.zero_grad()  # clear previous gradients

                with torch.cuda.amp.autocast():  # mixed precision training
                    outputs = self.model(inputs, masks)  # forward pass
                    loss = self.criterion(outputs, labels)  # calculate loss

                # Scale gradients
                self.scaler.scale(loss).backward()  # backward pass
                self.scaler.step(self.optimizer)  # update weights
                self.scaler.update()  # update scaler
                if use_scheduler:
                    scheduler.step()  # update learning rate

                train_loss += loss.item()  # add loss to train_loss
                # add accuracy to train_acc
                train_acc += (outputs.argmax(1) ==
                              labels).float().mean().item()

            # calculate average loss
            train_loss = train_loss / self.num_train_batches
            # calculate average accuracy
            train_acc = train_acc / self.num_train_batches

            self.losses['train'].append(train_loss)
            self.accuracies['train'].append(train_acc)

            val_loss, val_acc = self.evaluate()  # evaluate model on validation set

            # print losses and accuracies
            print(f'Epoch {epoch+1}/{num_epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

    def evaluate(self):
        self.model.eval()  # set model to evaluation mode
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for _, (inputs, masks, labels) in enumerate(self.val_data_loader):  # loop over batches
                inputs = inputs.to(self.device)  # move inputs to device
                masks = masks.to(self.device)  # move masks to device
                labels = labels.to(self.device)  # move labels to device

                outputs = self.model(inputs, masks)  # forward pass
                loss = self.criterion(outputs, labels)  # calculate loss

                val_loss += loss.item()  # add loss to val_loss
                # add accuracy to val_acc
                val_acc += (outputs.argmax(1) == labels).float().mean().item()

        # calculate average loss
        val_loss = val_loss / self.num_val_batches
        # calculate average accuracy
        val_acc = val_acc / self.num_val_batches

        self.losses['val'].append(val_loss)
        self.accuracies['val'].append(val_acc)

        self.model.train()  # set model back to training mode
        return val_loss, val_acc  # return losses and accuracies

    def test(self):
        # test the model on test set
        self.model.eval()  # set model to evaluation mode

        true_labels = []  # to store true labels
        pred_labels = []  # to store predicted labels
        with torch.no_grad():
            for _, (inputs, masks, labels) in enumerate(self.test_data_loader):  # loop over batches
                inputs = inputs.to(self.device)  # move inputs to device
                masks = masks.to(self.device)  # move masks to device
                labels = labels.to(self.device)  # move labels to device

                outputs = self.model(inputs, masks)  # forward pass

                # add true labels to true_labels
                true_labels.extend(labels.tolist())
                # add predicted labels to pred_labels
                pred_labels.extend(outputs.argmax(1).tolist())

        return true_labels, pred_labels  # return true and predicted labels
