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


def main(img_path):
    model = EyeGazeEstimatorModel()
    model = torch.nn.DataParallel(model)
    model.cuda()
    cudnn.benchmark = True

    saved = load_checkpoint()
    if saved:
        print('Loading checkpoint..... ')
        state = saved['state_dict']
        try:
            model.module.load_state_dict(state)
        except:
            model.load_state_dict(state)
    else:
        print('Warning: Could not read checkpoint!')
    time_start = time.time()
    imEyeL, imEyeR, leftEyeGrid, rightEyeGrid = load_data(img_path)
    print('load data time is : {running_time:.3f}'.format(running_time=time.time() - time_start))
    validate(model, imEyeL, imEyeR, leftEyeGrid, rightEyeGrid)
    print('total time is : {running_time:.3f}'.format(running_time=time.time() - time_start))
    return


def validate(model, imEyeL, imEyeR, leftEyeGrid, rightEyeGrid):
    # switch to evaluate mode
    model.eval()

    imEyeL = imEyeL.cuda(async=True)
    imEyeR = imEyeR.cuda(async=True)
    leftEyeGrid = leftEyeGrid.cuda(async=True)
    rightEyeGrid = rightEyeGrid.cuda(async=True)

    imEyeL = torch.autograd.Variable(imEyeL)
    imEyeR = torch.autograd.Variable(imEyeR)
    leftEyeGrid = torch.autograd.Variable(leftEyeGrid)
    rightEyeGrid = torch.autograd.Variable(rightEyeGrid)


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
    main("000005.png")
    print('DONE')
