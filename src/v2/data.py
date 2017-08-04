import math
import numpy as np
from skimage import io
from sklearn.preprocessing import scale


def normalize_image(input_image):
    input_image = input_image - input_image.mean()
    input_image = input_image / input_image.std()
    return input_image


class MnistDataLoader(object):
    def load_data(self, img):
        reshaped = img.copy().reshape((img.shape[0], 1, 28, 28))
        return reshaped

    def load_labels(self, label_ref):
        return label_ref


class NormalizingCropOrPadToSizeImageLoader(MnistDataLoader):
    def __init__(self, target_width, target_height, grayscale=True):
        self.target_width = target_width
        self.target_height = target_height
        self.grayscale = grayscale

    def load_data(self, image_location):
        readImage = io.imread(image_location)
        if len(readImage.shape) > 2:
            grayscaleImage = self.__rgb2gray(readImage)
        else:
            grayscaleImage = readImage
        final_image = self.__crop_and_or_pad_to_size(grayscaleImage, self.target_height, self.target_width)

        return np.array([[normalize_image(final_image)]])

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
        final_image = grayscaleImage[startH:endH, startW:endW]
        return final_image

    def __rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class NumpyArrayStftMagnitudeDataLoader(MnistDataLoader):
    def __init__(self, pca_model, label_loader, scale_features=False):
        super(NumpyArrayStftMagnitudeDataLoader, self).__init__()
        self.pca_freq = pca_model
        self.scale_features = scale_features
        self.label_loader = label_loader

    def load_data(self, batch_data_refs):
        batch = []
        for ref in batch_data_refs:
            magAndPhaseArr = np.load(ref)
            magArr = magAndPhaseArr[0].T
            if self.scale_features:
                magArr = scale(magArr)
            magArr = np.asarray([self.pca_freq.transform(magArr).T])
            batch.append(magArr.swapaxes(0, 1))

        """batch = np.asarray(batch)
        batch_shape = batch.shape
        batch = batch.reshape((batch_shape[0] * batch_shape[1], batch_shape[2]))
        if self.scale_features:
            batch = scale(batch)
        batch = self.pca_freq.transform(batch)
        batch = batch.reshape((batch_shape[0], 1, batch_shape[1], self.pca_freq.components_.shape[0]))"""

        return batch

    def load_labels(self, batch_refs):
        labels = []
        for label_ref in batch_refs:
            labels.append(self.label_loader.load_label(label_ref))
        return labels


class LabelResolver(object):
    def load_label(self, label_ref):
        raise NotImplementedError("Use child class")


class TimitGenderLabelResolver(LabelResolver):
    TIMIT_LABEL_MALE = 0
    TIMIT_LABEL_FEMALE = 1

    def load_label(self, label_ref):
        return self.TIMIT_LABEL_MALE if label_ref[label_ref.rfind('/DR') + 5:label_ref.rfind(
            '/DR') + 6] is 'M' else self.TIMIT_LABEL_FEMALE
