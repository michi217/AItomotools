from model import Resnet101
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchmetrics import Dice
#from utils import *

class Optimizer():

    @staticmethod
    def train(dataloader, params: dict):
        """
        Training loop for lung tumour segmentation
        """

        # Get network model
        model = Resnet101(pretrained=True).to(params['device'])

        # Define optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate']) 
        criterion = nn.BCEWithLogitsLoss()

        # Evaluation metrics
        dice = Dice().to(params['device'])

        # SummaryWriter to write to tensorboard
        writer = SummaryWriter('/store/DAMTP/ml2119/segmentation_results/runs/')

        # Training loop
        for epoch in range (params['epochs']):
            # Sums up loss for one epoch
            running_loss = 0.
            steps = 0
            comp_dice = 0.

            # Loop over whole dataset
            for idx, (reconstruction_tensor, mask_tensor) in enumerate(dataloader):

                # Zero gradient
                optimizer.zero_grad()

                # Make predictions for this batch
                pred = model(reconstruction_tensor.float().to(params['device']))

                # Binarise results
                pred[pred>=0.5] = 1
                pred[pred<0.5] = 0

                # Compute loss
                loss = criterion(pred, mask_tensor.float().to(params['device'])) 
                loss.backward()
                running_loss += loss.item()

                # Compute evaluation metrics
                comp_dice += dice(pred, mask_tensor.to(params['device']))

                # Adjust learning weights
                optimizer.step()

                if steps%100 == 0 and steps != 0:
                
                    # Print results for this epoch
                    print(f'Epoch: {steps}, Loss: {running_loss:.4f}, Dice: {comp_dice/steps}')   

                    # Save intermediate results for training in tensorboard
                    writer.add_scalar('Loss/train', running_loss, steps)
                    writer.add_scalar('Dice/train', comp_dice/steps, steps)

                    Optimizer.save_loss_values_in_txt('/store/DAMTP/ml2119/segmentation_results/' + params['timestr'] + '/', steps, running_loss)
                    Optimizer.save_metric_values_in_txt('/store/DAMTP/ml2119/segmentation_results/' + params['timestr'] + '/', steps, comp_dice.item()/steps)
                    
                    running_loss = 0.

                if steps%500 == 0:
                    model.save_model('/store/DAMTP/ml2119/segmentation_results/' + params['timestr'] + '/' +str(steps))

                steps+=1

        # Return of trained model
        return model



    @staticmethod
    def eval(model, valdataloader, params: dict, savepath: str):
        """
        Evaluation of given model with given valdataloader
        Resulting images get saved in savepath (but only segmentation mask frames that are not completely 
        black and do not contain any nodule)
        """
        
        # Variables for evaluation metrics
        running_loss = 0.
        running_dice = 0.

        steps = 0
        image_counter = 0

        # Define criterion and metric
        criterion = nn.BCEWithLogitsLoss()
        dice = Dice().to(params['device'])

        # SummaryWriter to write to tensorboard
        writer = SummaryWriter()

        # Model
        model.eval()

        # Loop over whole dataset
        for idx, (reconstruction_tensor, mask_tensor) in enumerate(valdataloader):
            with torch.no_grad():

                # Make predictions for this batch
                pred = model(reconstruction_tensor.float().to(params['device']))

                # Binarise results
                pred[pred>=0.5] = 1
                pred[pred<0.5] = 0

                # Compute loss
                loss = criterion(pred, mask_tensor.float().to(params['device'])) 
                running_loss += loss.item()

                # Compute metric
                running_dice += dice(pred, mask_tensor.to(params['device']))

                print(steps)

                # Visualise results and save .png (only for frames that are not completely black = frames that contain nodules)
                for index in range(len(pred)):
                    if 1 in mask_tensor[index, 1]: #or 1 in pred[index, 0]: # where mask_tensor has a values of 1 in their foreground (nodule) channel
                        Optimizer.save_image(pred[index][1].float().detach().cpu(), mask_tensor[index][1].float().detach().cpu(), savepath + str(image_counter) + '.png')
                        image_counter+=1
            
                # Increase counter
                steps+=1

        return running_loss, running_dice/steps


    @staticmethod
    def save_image(image_tensor, mask_tensor, savepath: str):
        """
        Save evaluation image prediction and target in .png file
        """

        fig = plt.figure()

        # Prediction plot
        fig.add_subplot(1, 2, 1).set_title('Prediction')   
        plt.imshow(image_tensor, cmap=plt.cm.gray)

        # Target plot
        fig.add_subplot(1, 2, 2).set_title('Target')   
        plt.imshow(mask_tensor, cmap=plt.cm.gray)
        
        # Save plot
        plt.savefig(savepath)

        plt.close()

    @staticmethod
    def save_loss_values_in_txt(savepath, epoch, loss_value):
        with open(savepath + '/loss.txt', 'a+') as f:
            f.write(str(epoch) + ' ' + str(round(loss_value, 2)) + '\n')

    
    @staticmethod
    def save_metric_values_in_txt(savepath, epoch, metric_value):
        with open(savepath + '/metric.txt', 'a+') as f:
            f.write(str(epoch) + ' ' + str(round(metric_value, 2)) + '\n')


