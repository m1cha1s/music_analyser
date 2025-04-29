import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNMusicRecogniser(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNMusicRecogniser, self).__init__()
        #1st convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #2nd convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #3rd conv block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        #128x128 input = map size 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, y=None):
        #conv block 1
        x = self.conv1(x) #applies first convolution
        x = F.relu(x) # ReLU function increases the complexity of the neural network by introducing non-linearity, which allows the network to learn more complex representations of the data. The ReLU function is defined as f(x) = max(0, x), which sets all negative values to zero.
        x = self.pool(x) # reduces size using max pooling, (if we have 4x4 block of values, it makes 2x2 block where for every 2x2 array from 4x4 block takes the highest value)
        x = self.bn1(x) #normalizes the output

        #conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.bn2(x)

        #conv block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.bn3(x)

        #flattenign the tensor
        x = x.view(x.size(0), -1)

        #connect the layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

#        print(x)

        x = F.softmax(x, dim=-1)

        if y is None:
            loss = None
        else:
            loss_mlp = nn.BCELoss()
            loss = loss_mlp(x, y)

        return x, loss

