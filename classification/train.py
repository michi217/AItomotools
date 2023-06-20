from model import Resnet101
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy

class Optimizer():

    @staticmethod
    def train(dataloader, params: dict):
        """
        Training loop for lung tumour classification
        """

        # Get network model
        model = Resnet101(pretrained=True).to(params['device'])

        # Define optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate']) 
        criterion = nn.CrossEntropyLoss()

        # Evaluation metrics
        accuracy = MulticlassAccuracy(num_classes=4).to(params['device'])

        # SummaryWriter to write to tensorboard
        writer = SummaryWriter('/store/DAMTP/ml2119/classification_results/runs/')

        # Training loop
        for epoch in range (params['epochs']):
            # Sums up loss for one epoch
            running_loss = 0.
            running_accuracy = 0.
            steps = 0

            # Loop over whole dataset
            for idx, (reconstruction_tensor, diagnosis) in enumerate(dataloader):

                # Zero gradient
                optimizer.zero_grad()

                # Make predictions for this batch
                pred = model(reconstruction_tensor.float().to(params['device']))

                # Compute loss
                loss = criterion(pred, diagnosis.to(params['device'])) 
                loss.backward()
                running_loss += loss.item()

                # Compute evaluation metrics
                running_accuracy += accuracy(pred, diagnosis.to(params['device']))

                # Adjust learning weights
                optimizer.step()

                if steps%100 == 0 and steps != 0:
                
                    # Print results for this epoch
                    print(f'Epoch: {steps}, Loss: {running_loss:.4f}, Accuracy: {running_accuracy/steps}')    

                    # Save intermediate results for training in tensorboard
                    writer.add_scalar('Loss/train', running_loss, steps)
                    writer.add_scalar('Accuracy/train', running_accuracy/steps, steps)
                    
                    Optimizer.save_loss_values_in_txt('/store/DAMTP/ml2119/classification_results/' + params['timestr'] + '/', steps, running_loss)
                    Optimizer.save_metric_values_in_txt('/store/DAMTP/ml2119/classification_results/' + params['timestr'] + '/', steps, running_accuracy.item()/steps)

                    running_loss = 0.

                if steps%500 == 0:
                    model.save_model('/store/DAMTP/ml2119/classification_results/' + params['timestr'] + '/' +str(steps))

                steps+=1

        # Return of trained model
        return model
    

    @staticmethod
    def eval (model, valdataloader, params: dict, savepath: str):
        """
        Evaluation of given model with given valdataloader
        Resulting images get saved in savepath
        """
            
        # Variables for evaluation metrics
        running_loss = 0.
        running_accuracy = 0.
        
        steps = 0
        image_counter = 0
        
        # Define criterion and metric
        criterion = nn.CrossEntropyLoss()
        accuracy = MulticlassAccuracy(num_classes=4).to(params['device'])

        # SummaryWriter to write to tensorboard
        writer = SummaryWriter()

        # Model
        model.eval()

        # Loop over whole dataset
        for idx, (reconstruction_tensor, diagnosis) in enumerate(valdataloader):
            with torch.no_grad():

                # Make predictions for this batch
                pred = model(reconstruction_tensor.float().to(params['device']))

                # Compute loss
                loss = criterion(pred, diagnosis.to(params['device'])) 
                running_loss += loss.item()

                # Compute metric
                running_accuracy += accuracy(pred, diagnosis.to(params['device']))

                print(steps)

                pred_tensor = pred.argmax(axis=-1)

                for index in range(len(pred)):
                    if diagnosis[index] != 0:
                        Optimizer.save_image(reconstruction_tensor[index][0].float().detach().cpu(), pred_tensor[index].item(), diagnosis[index].item(), savepath + str(image_counter) + '.png')
                        image_counter+=1

                # Increase counter
                steps+=1

        return running_loss, running_accuracy/steps


    @staticmethod
    def save_image(image_tensor, pred_tensor, diagnosis_tensor, savepath: str):
        """
        Save evaluation image prediction and target in .png file
        """

        fig = plt.figure()

        # Figure plot
        fig.suptitle('Target: ' + str(diagnosis_tensor) + ' , Prediction: ' + str(pred_tensor))  
        plt.imshow(image_tensor, cmap=plt.cm.gray_r)
        
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
