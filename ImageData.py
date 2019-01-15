import os, math
import scipy.io as sio
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import face_alignment
import torch

MEAN_PATH = './data/'
gridSize = [50, 50]

def loadMetadata(filename, silent=False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


def get_image_dim(image_name):
    image = cv2.imread(image_name)
    height, width = image.shape[:2]
    return height, width


def load_eye_image(image, imSize=(224, 224), left=True):
    """load image, returns cuda tensor"""
    if left:
        eyeMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_left_224.mat'))['image_mean']
    else:
        eyeMean = loadMetadata(os.path.join(MEAN_PATH, 'mean_right_224.mat'))['image_mean']

    transformEye = transforms.Compose([
        transforms.Resize(imSize),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    imEye = transformEye(image)

    trans_gray = transforms.Grayscale(num_output_channels=1)
    eye_mean = trans_gray(Image.fromarray(np.uint8(eyeMean)))
    meanImg = transforms.ToTensor()(eye_mean)
    imEye = imEye.sub(meanImg)
    return imEye


def cutArea(image, x_list, y_list, offset=0.5):
    height, width = image.shape[:2]

    top = int(min(y_list))
    bottom = int(max(y_list))
    left = int(min(x_list))
    right = int(max(x_list))

    offset_x = int(math.ceil(abs(right - left) * offset))
    offset_y = int(math.ceil(abs(bottom - top) * offset))

    center_x = int((left + right) / 2)
    center_y = int((top + bottom) / 2)

    offset = offset_x + int(math.ceil(abs(right - left) / 2)) if offset_x > offset_y else offset_y + int(
        math.ceil(abs(bottom - top) / 2))

    # top, bottom, left, right
    return max(0, center_y - offset), min(height - 1, center_y + offset), max(0, center_x - offset), min(width - 1,
                                                                                                         center_x + offset)

def extractEyeArea(image, landmarks):
    r_pointX = []
    r_pointY = []
    l_pointX = []
    l_pointY = []
    height, width = image.shape[:2]

    for i in range(len(landmarks)):
        # right eye
        if i >= 36 and i <= 41:
            r_pointX.append(landmarks[i][0])
            r_pointY.append(landmarks[i][1])

        # left eye
        elif i >= 42 and i <= 47:
            l_pointX.append(landmarks[i][0])
            l_pointY.append(landmarks[i][1])

    r_top, r_bottom, r_left, r_right = cutArea(image, x_list=r_pointX, y_list=r_pointY)
    if r_top >= 0 and r_left >= 0 and r_bottom < height and r_right < width:
        r_eye_img = image[r_top:r_bottom, r_left:r_right]

    l_top, l_bottom, l_left, l_right = cutArea(image, x_list=l_pointX, y_list=l_pointY)
    if l_top >= 0 and l_left >= 0 and l_bottom < height and l_right < width:
        l_eye_img = image[l_top:l_bottom, l_left:l_right]

    return r_eye_img, l_eye_img, [r_top, r_bottom, r_left, r_right], [l_top, l_bottom, l_left, l_right]


def find_and_save_eyes(image_name):
    if not os.path.isfile(image_name):
        return None, None, None, None

    image = cv2.imread(image_name)
    if image is None:
        return None, None, None, None

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)
    face_landmarks_list = fa.get_landmarks(image)
    if face_landmarks_list == None:
        return None, None, None, None

    face_landmarks_list = face_landmarks_list[0]

    r_eye_img, l_eye_img, r_box, l_box = extractEyeArea(image, face_landmarks_list)

    return r_eye_img, l_eye_img, r_box, l_box


def makeGrid(gridSize, params):
    gridLen = gridSize[0] * gridSize[1]
    grid = np.zeros([gridLen, ], np.float32)

    indsY = np.array([i // gridSize[0] for i in range(gridLen)])
    indsX = np.array([i % gridSize[0] for i in range(gridLen)])
    condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
    condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
    cond = np.logical_and(condX, condY)

    grid[cond] = 1
    return grid


def get_eye_grid(frameW, frameH, gridW, gridH, labelFaceX, labelFaceY, labelFaceW, labelFaceH):
    scaleX = gridW / frameW
    scaleY = gridH / frameH

    xLo = round(labelFaceX * scaleX) + 1
    yLo = round(labelFaceY * scaleY) + 1
    w = round(labelFaceW * scaleX)
    h = round(labelFaceH * scaleY)
    gridPara = [xLo, yLo, w, h]

    grid = makeGrid([50,50], gridPara)
    return grid


def load_data(image_name):
    #Image.fromarray(np.uint8(meanImg)
    height, width = get_image_dim(image_name)
    r_eye_img, l_eye_img, r_box, l_box = find_and_save_eyes(image_name)
    r_img = load_eye_image(Image.fromarray(np.uint8(r_eye_img)), left=False)
    l_img = load_eye_image(Image.fromarray(np.uint8(l_eye_img)), left=True)

    l_eye_grid = get_eye_grid(width, height, gridSize[0], gridSize[1], l_box[2], l_box[0], l_box[3]-l_box[2]+1, l_box[1]-l_box[0]+1)
    l_eye_grid = torch.FloatTensor(l_eye_grid)
    r_eye_grid = get_eye_grid(width, height, gridSize[0], gridSize[1], r_box[2], r_box[0], r_box[3]-r_box[2]+1, r_box[1]-r_box[0]+1)
    r_eye_grid = torch.FloatTensor(r_eye_grid)
    return l_img.unsqueeze_(0), r_img.unsqueeze_(0), l_eye_grid.unsqueeze_(0), r_eye_grid.unsqueeze_(0)
