import os

from torchvision import models
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2 as cv

FILE_NAME_NUM_DIGITS = 6  # number of digits in the frame/mask names, e.g. for 6: '000023.jpg'
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4']

PERSON_CHANNEL_INDEX = 15  # segmentation stage

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CUDA_EXCEPTION_CODE = 1
ERROR_CODE = 1


def post_process_mask(mask):
    """
    Helper function for automatic mask (produced by the segmentation model) cleaning using heuristics.
    """

    # step1: morphological filtering (helps splitting parts that don't belong to the person blob)
    kernel = np.ones((13, 13), np.uint8)  # hardcoded 13 simply gave nice results
    opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # step2: isolate the person component (biggest component after background)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(opened_mask)

    if num_labels > 1:
        # step2.1: find the background component
        h, _ = labels.shape  # get mask height
        # find the most common index in the upper 10% of the image - I consider that to be the background index (heuristic)
        discriminant_subspace = labels[:int(h/10), :]
        bkg_index = np.argmax(np.bincount(discriminant_subspace.flatten()))

        # step2.2: biggest component after background is person (that's a highly probable hypothesis)
        blob_areas = []
        for i in range(0, num_labels):
            blob_areas.append(stats[i, cv.CC_STAT_AREA])
        blob_areas = list(zip(range(len(blob_areas)), blob_areas))
        blob_areas.sort(key=lambda tup: tup[1], reverse=True)  # sort from biggest to smallest area components
        blob_areas = [a for a in blob_areas if a[0] != bkg_index]  # remove background component
        person_index = blob_areas[0][0]  # biggest component that is not background is presumably person
        processed_mask = np.uint8((labels == person_index) * 255)

        return processed_mask
    else:  # only 1 component found (probably background) we don't need further processing
        return opened_mask

import utils
def segment_photo(photo):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # img = cv.imread(os.getcwd()+"/data/video/"+"out58.png")
    img = photo
    transform = transforms.Compose([
            # transforms.Resize((segmentation_mask_height, segmentation_mask_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
        ])
    real_img = transform(img)


    def image_loader_tensor(image_tensor):
        # image_tensor = image_tensor.transpose((2, 0, 1))
        return image_tensor.unsqueeze(0).to(device, torch.float)


    segmentation_model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()
    result_batch = segmentation_model(image_loader_tensor(real_img))['out'].to('cpu').detach().numpy()  # shape: (N, 21, H, W) (21 - PASCAL VOC classes)
    mask = np.argmax(result_batch[0], axis=0) == PERSON_CHANNEL_INDEX
    mask = np.uint8(mask * 255)  # convert from bool to [0, 255] black & white image
    processed_mask = post_process_mask(mask)  # simple heuristics (connected components, etc.)
    # cv.imwrite("maska3.png", mask)
    return mask

