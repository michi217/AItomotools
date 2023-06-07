import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
import torchvision.models as models

class Resnet101(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()

        # if pretrained:
        #     # Get pretrained model from pytorch
        #     weights = FCN_ResNet101_Weights.DEFAULT.DEFAULT
        #     resnet101 = fcn_resnet101(weights=weights, num_classes=2)

        resnet101 = fcn_resnet101(num_classes=2)
        resnet101.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize network
        self.net = resnet101

    def forward(self, x):
        """
        Forward path of Resnet101
        """
        x = self.net(x)
        x = torch.sigmoid(x['out']) # Use sigmoid function to map input to values between 0 and 1
        return x
    
    def save_model(self, timestr):
        """
        Save trained model with given timestring or name
        """
        torch.save(self.state_dict(), 'model' + timestr + '.pth')