class LayerUnits(object):
    def __init__(self, shape):
        super(LayerUnits, self).__init__()
        self.__shape = shape
        self.__connected_up = None
        self.__connected_down = None
        self.__value = None

    def get_shape(self):
        return self.__shape

    def set_connected_up(self, layer):
        self.__connected_up = layer

    def get_connected_up(self):
        return self.__connected_up

    def set_connected_down(self, layer):
        self.__connected_down = layer

    def get_connected_down(self):
        return self.__connected_down

    def set_value(self, value):
        self.__value = value

    def get_value(self):
        return self.__value
