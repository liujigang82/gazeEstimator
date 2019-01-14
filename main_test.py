import os, time
import numpy as np
import torch

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import scipy.io as sio

from EyeGazeEstimatorModel import EyeGazeEstimatorModel
from ImageData import load_data

'''
Train/test code for Eye Gaze Tracker.

Author: Liu Jigang ( liujg@ntu.edu.sg). 

'''


def main():
    global args, best_prec1, weight_decay, momentum

    model = EyeGazeEstimatorModel()
    model = torch.nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    saved = load_checkpoint()
    if saved:
        print('Loading checkpoint for epoch %05d with loss %.5f (which is L2 = mean of squares)...' % (saved['epoch']
                                                                                                           , saved['best_prec1']))
        state = saved['state_dict']
        try:
            model.module.load_state_dict(state)
        except:
            model.load_state_dict(state)
    else:
        print('Warning: Could not read checkpoint!')

    imEyeL, imEyeR, leftEyeGrid, rightEyeGrid = load_data("/home/jigang/data/GazeCaputreData/00006/frames/00000.jpg")
    print(imEyeL.shape, imEyeR.shape, leftEyeGrid.shape, rightEyeGrid.shape)

    validate(model, imEyeL, imEyeR, leftEyeGrid, rightEyeGrid)
    return


def validate(model, imEyeL, imEyeR, leftEyeGrid, rightEyeGrid):
    # switch to evaluate mode
    model.eval()
    output = model(imEyeL, imEyeR, leftEyeGrid, rightEyeGrid)
    print(output)
    return output

CHECKPOINTS_PATH = '.'

def load_checkpoint(filename='checkpoint.pth.tar'):

    filename = 'best_checkpoint.pth.tar'
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename)
    return state


if __name__ == "__main__":
    main()
    print('DONE')
