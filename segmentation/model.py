import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights

class Resnet101(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()

        if pretrained:
            # Get pretrained model from pytorch
            weights = FCN_ResNet101_Weights.DEFAULT.DEFAULT
            resnet101 = fcn_resnet101(weights=weights)

        else: resnet101 = fcn_resnet101()

        # Initialize network
        self.net= resnet101

    def forward(self, x):
        """
        Forward path of Resnet101
        """
        x = self.net(x)
        return x
    
    def save_model(self, timestr):
        """
        Save trained model with given timestring or name
        """
        torch.save(self.state_dict(), 'model' + timestr + '.pth')