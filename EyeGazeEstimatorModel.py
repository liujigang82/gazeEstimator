from resnet import resnet18
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

'''

Train/test code for Eye Gaze Tracker.

Author: Liu Jigang ( liujg@ntu.edu.sg). 
'''


class ImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ImageModel, self).__init__()
        self.features = resnet18(input_channel=1, num_classes=256)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class LeftEyeModel(nn.Module):
    def __init__(self):
        super(LeftEyeModel, self).__init__()
        self.conv = ImageModel()
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        print("x size:", x.shape)
        x = self.fc(x)
        return x


class RightEyeModel(nn.Module):
    def __init__(self):
        super(RightEyeModel, self).__init__()
        self.conv = ImageModel()
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class EyeGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize =50):
        super(EyeGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EyeGazeEstimatorModel(nn.Module):

    def __init__(self):
        super(EyeGazeEstimatorModel, self).__init__()
        self.leftEyeModel = LeftEyeModel()
        self.leftGrid = EyeGridModel()
        self.rightEyeModel = RightEyeModel()
        self.rightGrid = EyeGridModel()
        # Joining both eyes

        self.eyesFC = nn.Sequential(
            nn.Linear(64+128, 128),
            nn.ReLU(inplace=True),
            )

        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128+128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            )

    def forward(self, eyesLeft, eyesRight, leftGrids, rightGrids):
        # Eye nets
        xEyeL = self.leftEyeModel(eyesLeft)
        xLeftGrid = self.leftGrid(leftGrids)
        xLeftEye = torch.cat((xEyeL, xLeftGrid), 1)
        xLeftEye = self.eyesFC(xLeftEye)

        xEyeR = self.rightEyeModel(eyesRight)
        xRightGrid = self.rightGrid(rightGrids)

        xRightEye = torch.cat((xEyeR, xRightGrid), 1)
        xRightEye = self.eyesFC(xRightEye)

        # Cat all
        x = torch.cat((xLeftEye, xRightEye), 1)
        x = self.fc(x)
        
        return x
