from turtle import forward
from numpy import pad
import torch
import torch.nn as nn

'''
INPUT 32 x 32
Convolutions - Subsampling - Convolution - Subsampling - Full connection - Full connection - Gaussian connections


'''
class Lenet5(nn.Module):
    def __init__(self, batch, n_classes, in_channel, in_width, in_height, is_train = False):
        super().__init__()
        self.batch = batch
        self.n_classes = n_classes
        self.in_width = in_width
        self.in_height = in_height
        self.in_channel = in_channel
        self.is_train = is_train 

        # nn.Conv2d(num_channel_size, output_channel_size, kerner_size, stride, padding)
        # output_channel_size -> number of filers(kernels) 
        # 5 filters yields 5 results(images) hence, 5 channels

        # convolution output : [(W - K + 2P)/S] + 1

        # Convolution [(32 - 5 + 2*0) / 1] + 1 = 28
        self.conv0 = nn.Conv2d(self.in_channel, 6, kernel_size=5, stride=1, padding=0)

        # Average pooling [(28 - 2 + 2*0)/2] + 1 = 26/2 + 1 = 14 
        
        # Convolution [(14 - 5 + 2*0) / 1] + 1 = 9 + 1 = 10 
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

        # Avearge pooling [(10 - 2 + 2*0)/2] + 1 = 5
        
        # Convolution [(14 - 5 + 2*0) / 1] + 1 = 10
        self.conv2 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)

        # Average Pooling fully-connected layer
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, self.n_classes)

    def forward(self, x):
        # x' shape : [B, C, H, W]
        x = self.conv0(x)        
        x = torch.tanh(x) # activation function
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv1(x)
        x = torch.tanh(x)        
        x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = torch.tanh(x)

        # change format from 4dim -> 2dim ( [B, C, H, W] -> [B, C*H*W])
        x = torch.flatten(x, start_dim=1)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc4(x)
        x = x.view(self.batch, -1)
        x = nn.functional.softmax(x, dim=1)

        if self.is_train is False:
            x = torch.argmax(x, dim=1)
        return x

        
        
       
        
        

        