from model import Resnet101
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt


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
        for epoch in range (params['epochs']):
            # Sums up loss for one epoch
            running_loss = 0.
            steps = 0

            # Loop over whole dataset
            for idx, (reconstruction_tensor, mask_tensor) in enumerate(dataloader):

                # Zero gradient
                optimizer.zero_grad()

                # Make predictions for this batch
                pred = model(reconstruction_tensor.float().to(params['device']))

                # Compute loss
                loss = criterion(pred, mask_tensor.float().to(params['device'])) 
                loss.backward()
                running_loss += loss.item()

                # Adjust learning weights
                optimizer.step()

                if steps%50 == 0:
                
                    # Print results for this epoch
                    print(f'Epoch: {steps}, Loss: {running_loss:.2f}')    

                    # Save intermediate results for training in tensorboard
                    writer.add_scalar('Loss/train', running_loss, epoch)
                    
                    running_loss = 0.

                if steps%100 == 0:
                    # Binarize segmentation mask for output
                    pred[pred>=0.5] = 1
                    pred[pred<0.5] = 0

                    fig=plt.figure()

                    fig.add_subplot(1, 2, 1).set_title('Prediction')   
                    plt.imshow(pred[0][0].float().detach().cpu(), cmap=plt.cm.gray)#, cmap=cmap)

                    fig.add_subplot(1, 2, 2).set_title('Target')   
                    # my data is OK to use gray colormap (0:black, 1:white)
                    plt.imshow(mask_tensor[0][0].float().detach().cpu(), cmap=plt.cm.gray)#, cmap=plt.cm.gray)  
                    plt.savefig('result' + str(steps) + '.png')

                steps+=1

                if steps == 1000:
                    break

        # Return of trained model
        return model

