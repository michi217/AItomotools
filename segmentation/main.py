import torch
from torch.utils.data import DataLoader 
import time
from train import Optimizer
from AItomotools.data_loaders.LIDC_IDRI import LIDC_IDRI
import os

class Segmentation():
    def __init__(self, data, valdata, parameters):
        """
        Initialisation of parameters for segmentation algorithm
        """
        self.data = data
        self.valdata = valdata
        self.params = parameters

    def __call__(self):
        """
        Run segmentation algorithm with evaluation
        """

        # Create folders to save run
        if not os.path.exists('/store/DAMTP/ml2119/segmentation_results/' + self.params['timestr']):
            os.makedirs('/store/DAMTP/ml2119/segmentation_results/' + self.params['timestr'])
            os.makedirs('/store/DAMTP/ml2119/segmentation_results/' + self.params['timestr'] + '/validationimages')

        # Create dataloader
        dataloader = DataLoader(self.data, batch_size=self.params['batch_size'], shuffle=True)
        valdataloader = DataLoader(self.valdata, batch_size=self.params['batch_size'], shuffle=False)

        # Run training loop
        optimizing = Optimizer()
        model = optimizing.train(dataloader, self.params)
        
        # Save model
        model.save_model('/store/DAMTP/ml2119/segmentation_results/' + self.params['timestr'] + '/')

        # Evaluation
        #model = Resnet101(pretrained=True)
        #model.load_state_dict(torch.load("model07062023-1243.pth"))
        model.eval()
        loss, dice = optimizing.eval(model, valdataloader, self.params, '/store/DAMTP/ml2119/segmentation_results/' + self.params['timestr'] + '/validationimages/')
        print(str(loss) + ', ' + str(dice))


        # Save training parameters
        with open('/store/DAMTP/ml2119/segmentation_results/' + self.params['timestr'] + '/parameters.txt', 'w') as f:
            print(self.params, file=f)

def main():
    # Create a time string to identify the run
    timestr = time.strftime("%d%m%Y-%H%M")

    # Initialize dataset
    data = LIDC_IDRI("segmentation", 0.8, "training")
    valdata = LIDC_IDRI("segmentation", 0.8, "testing")

    print(str(len(data)) + ' datapoints for training and ' + str(len(valdata)) + ' datapoints for testing.')

    # Set training parameters
    params = {
        "device": torch.device('cuda:3'),
        "learning_rate": 0.00001,
        "epochs": 1,
        "batch_size": 8,
        "momentum": None,
        "weigth_decay": None,
        "timestr": timestr,
    }

    # Initialise Segmentation
    segmentation = Segmentation(data, valdata, params)

    # Run algorithm
    segmentation()

main()
