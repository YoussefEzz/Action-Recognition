import torch.nn as nn
import torch.nn.functional

class ActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1, padding=1)
 
        self.fc3 = nn.Linear(64 * 5 * 5, 128)
 
        self.fc4 = nn.Linear(128, 40)
 
    def forward(self, x):
        # input 3x128x128, output 32x128x128
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x,2)

        # input 32x128x128, output 64x128x128
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)

        x = x.view(-1, 64 * 5 * 5)
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x