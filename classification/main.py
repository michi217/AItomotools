import torch
from torch.utils.data import DataLoader 
import time
from train import Optimizer
from AItomotools.data_loaders.LIDC_IDRI import LIDC_IDRI
import os

class Classification():
    def __init__(self, data, valdata, parameters):
        """
        Initialisation of parameters for classification algorithm
        """
        self.data = data
        self.valdata = valdata
        self.params = parameters

    def __call__(self):
        """
        Run classification algorithm with evaluation
        """

        # Create folders to save run
        if not os.path.exists('/store/DAMTP/ml2119/classification_results/' + self.params['timestr']):
            os.makedirs('/store/DAMTP/ml2119/classification_results/' + self.params['timestr'])
            os.makedirs('/store/DAMTP/ml2119/classification_results/' + self.params['timestr'] + '/validationimages')


        # Create dataloader
        dataloader = DataLoader(self.data, batch_size=self.params['batch_size'], shuffle=True)
        valdataloader = DataLoader(self.valdata, batch_size=self.params['batch_size'], shuffle=False)

        # Run training loop
        optimizing = Optimizer()
        model = optimizing.train(dataloader, self.params)
        
        # Save model
        model.save_model('/store/DAMTP/ml2119/classification_results/' + self.params['timestr'] + '/')

        # # Evaluation
        # model.eval()
        loss, accuracy = optimizing.eval(model, valdataloader, self.params, '/store/DAMTP/ml2119/classification_results/' + self.params['timestr'] + '/validationimages/')
        print(str(loss) + ', ' + str(accuracy))
        
        # Save training parameters
        with open('/store/DAMTP/ml2119/classification_results/' + self.params['timestr'] + '/parameters.txt', 'w') as f:
            print(self.params, file=f)

def main():
    # Create a time string to identify the run
    timestr = time.strftime("%d%m%Y-%H%M")

    # Initialize dataset
    data = LIDC_IDRI("diagnostic", 0.8, "training")
    valdata = LIDC_IDRI("diagnostic", 0.8, "testing")

    # Set training parameters
    params = {
        "device": torch.device('cuda:2'),
        "learning_rate": 0.00001,
        "epochs": 10,
        "batch_size": 8,
        "momentum": None,
        "weigth_decay": None,
        "timestr": timestr,
        "classes": ('class 0', 'class 1', 'class 2', 'class 3')
    }

    # Initialize Segmentation
    classification = Classification(data, valdata, params)

    # Run algorithm
    classification()

main()
