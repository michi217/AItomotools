import torch
from torch.utils.data import DataLoader 
import time
from train import Optimizer
#from AItomotools.utils.paths import LIDC_IDRI_PROCESSED_DATASET_PATH
from AItomotools.data_loaders.LIDC_IDRI import LIDC_IDRI

class Segmentation():
    def __init__(self, data, parameters):
        self.data = data
        self.params = parameters

    def __call__(self):
        """Run Algorithm"""
        # Create a time string to identify the run
        timestr = time.strftime("%d%m%Y-%H%M")

        # Create dataloader
        dataloader = DataLoader(self.data, batch_size=self.params['batch_size'], shuffle=True)

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
    data = LIDC_IDRI("segmentation", 0.8, "training")

    # Set training parameters
    params = {
        "device": torch.device('cuda'),
        "learning_rate": 0.001,
        "epochs": 1,
        "batch_size": 8,
        "momentum": None,
        "weigth_decay": None,
    }

    # Initialize Segmentation
    segmentation = Segmentation(data, params)

    # Run algorithm
    segmentation()

main()
