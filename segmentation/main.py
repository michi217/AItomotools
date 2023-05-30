import torch
from torch.utils.data import DataLoader 
import time
from train import Optimizer

class Segmentation():
    def __init__(self, data, parameters):
        self.data = data
        self.params = parameters

    def __call__(self):
        """Run Algorithm"""
        print(torch.cuda.is_available())
        # Create a time string to identify the run
        timestr = time.strftime("%d%m%Y-%H%M")

        # Create dataloader
        # TODO
        dataloader = None

        # Run training loop
        optimizing = Optimizer()
        model = optimizing.train(dataloader, self.params)
        
        # Save model
        model.save_model(timestr)

        # Save training parameters
        with open(timestr + '.txt', 'w') as f:
            print(self.params, file=f)

def main():
    # Initialize dataset
    data = None #TODO

    # Set training parameters
    params = {
        "device": torch.device('cuda'),
        "learning_rate": 0.001,
        "epochs": 2,
        "batch_size": 8,
        "momentum": None,
        "weigth_decay": None,
    }

    # Initialize Segmentation
    segmentation = Segmentation(data, params)

    # Run algorithm
    segmentation()

main()
