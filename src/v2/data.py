class MnistDataLoader(object):
    def load_data(self, img):
        return img.copy().reshape((1, 1, 28, 28))
