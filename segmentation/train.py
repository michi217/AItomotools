from model import Resnet101
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Optimizer():

    @staticmethod
    def train(dataloader, params: dict):

        # Get network model
        model = Resnet101(pretrained=True).to(params['device'])

        # Define optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate']) 
        criterion = nn.CrossEntropyLoss()

        # SummaryWriter to write to tensorboard
        writer = SummaryWriter()

        # Training loop
        # for epoch in range (params['epochs']):
        #     # Sums up loss for one epoch
        #     running_loss = 0.

        #     # Loop over whole dataset
        #     for idx, data in enumerate(dataloader):

        #         # TODO
        #         input, gt = None

        #         # Zero gradient
        #         optimizer.zero_grad()

        #         # Make predictions for this batch
        #         pred = model(input)

        #         # Compute loss
        #         loss = criterion(pred, gt) 
        #         loss.backward()
        #         running_loss += loss.item()

        #         # Adjust learning weights
        #         optimizer.step()
        
        #     # Print results for this epoch
        #     print(f'Epoch: {epoch}, Loss: {running_loss:.2f}')

        #     # Save intermediate results for training in tensorboard
        #     writer.add_scalar('Loss/train', running_loss, epoch)

        # Return of trained model
        return model

