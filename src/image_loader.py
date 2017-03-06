import math
import numpy as np
from skimage import io


class NormalizingResizingImageLoader(object):
    def __init__(self, target_width, target_height, grayscale=True):
        self.target_width = target_width
        self.target_height = target_height
        self.grayscale = grayscale

    def load_image(self, image_location):
        print image_location
        readImage = io.imread(image_location)
        if(len(readImage.shape) > 2):
            grayscaleImage = self.__rgb2gray(readImage)
        else:
            grayscaleImage = readImage
        final_image = self.__crop_and_or_pad_to_size(grayscaleImage, self.target_height, self.target_width)

        return np.array([[self.__normalize_image(final_image)]])

    def __crop_and_or_pad_to_size(self, grayscaleImage, target_height, target_width):
        diff_w = target_width - grayscaleImage.shape[1]
        diff_h = target_height - grayscaleImage.shape[0]
        # first pad the image if needed
        if diff_w > 0:
            padBefore = int(math.ceil(diff_w / 2.))
            padAfter = int(math.floor(diff_w / 2.))
            grayscaleImage = np.pad(grayscaleImage, [(0, 0), (padBefore, padAfter)], mode='constant', constant_values=0)
            startW = 0
            endW = grayscaleImage.shape[1]
        else:
            diff_w = -diff_w
            startW = int(math.ceil(diff_w / 2.))
            endW = grayscaleImage.shape[1] - int(math.floor(diff_w / 2.))
        if diff_h > 0:
            padBefore = int(math.ceil(diff_h / 2.))
            padAfter = int(math.floor(diff_h / 2.))
            grayscaleImage = np.pad(grayscaleImage, [(padBefore, padAfter), (0, 0)], mode='constant', constant_values=0)
            startH = 0
            endH = grayscaleImage.shape[0]
        else:
            diff_h = -diff_h
            startH = int(math.ceil(diff_h / 2.))
            endH = grayscaleImage.shape[0] - int(math.floor(diff_h / 2.))
        final_image = grayscaleImage[startH:endH,startW:endW]
        return final_image

    def __rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def __normalize_image(self, input_image):
        input_image = input_image - input_image.mean()
        input_image = input_image / input_image.std()
        return input_image
