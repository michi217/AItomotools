import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights


class Resnet101(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()

        # Initialize network
        resnet = resnet101()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net = nn.Sequential(resnet, nn.Linear(1000,4))

    def forward(self, x):
        """
        Forward path of Resnet101
        """
        x = self.net(x)
        return x
    
    def save_model(self, savepath):
        """
        Save trained model with given savepath
        """
        torch.save(self.state_dict(), savepath + 'classification_model.pth')