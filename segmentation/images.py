import torch
from AItomotools.data_loaders.LIDC_IDRI import LIDC_IDRI
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
from model import Resnet101

model = Resnet101(pretrained=True)
model.load_state_dict(torch.load("model05062023-1341.pth"))


data = LIDC_IDRI("segmentation", 0.8, "training")
dataloader = DataLoader(data, batch_size=8, shuffle=True)

for idx, (reconstruction_tensor, mask_tensor) in enumerate(dataloader):

    pred = model(reconstruction_tensor.float())

    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0

    for index in range(8):

        fig = plt.figure()

        fig.add_subplot(1, 2, 1).set_title('Prediction')   
        plt.imshow(pred[index][0].float().detach().cpu(), cmap=plt.cm.gray_r, vmin=0., vmax=1.)

        fig.add_subplot(1, 2, 2).set_title('Target')   
        plt.imshow(mask_tensor[index][0].float().detach().cpu(), cmap=plt.cm.gray_r, vmin=0., vmax=1.)
        plt.savefig('image' + str(index) + '.png')
        plt.close()
    break