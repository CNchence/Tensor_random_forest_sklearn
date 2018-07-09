import numpy
import cv2


class NPDFeature():
    """It is a tool class to extract the NPD features.
    Attributes:
        image: A two-dimension ndarray indicating grayscale image.
        n_pixels: An integer indicating the number of image total pixels.
        features: A one-dimension ndarray to store the extracted NPD features.
    """
    __NPD_table__ = None

    def __init__(self, image):
        '''Initialize NPDFeature class with an image.'''
        if NPDFeature.__NPD_table__ is None:
            NPDFeature.__NPD_table__ = NPDFeature.__calculate_NPD_table()
        # 断言确定是否为numpy数组
        assert isinstance(image, numpy.ndarray)
        # 返回视图
        self.image = image.ravel()
        self.n_pixels = image.size
        self.features = numpy.empty(shape=self.n_pixels * (self.n_pixels - 1) // 2, dtype=float)

    def extract(self):
        '''Extract features from given image.
        Returns:
            A one-dimension ndarray to store the extracted NPD features.
        '''
        count = 0
        for i in range(self.n_pixels - 1):
            for j in range(i + 1, self.n_pixels, 1):
                self.features[count] = NPDFeature.__NPD_table__[self.image[i]][self.image[j]]
                count += 1
        return self.features

    @staticmethod
    def __calculate_NPD_table():
        '''Calculate all situations table to accelerate feature extracting.'''
        # print("Calculating the NPD table...")
        table = numpy.empty(shape=(1 << 8, 1 << 8), dtype=float)
        for i in range(1 << 8):
            for j in range(1 << 8):
                if i == 0 and j == 0:
                    table[i][j] = 0
                else:
                    table[i][j] = (i - j) / (i + j)
        return table

if __name__ == '__main__':
    im = cv2.imread("L:\\Dataset\\Tensor\\1529592066\\image\\1529592068.bmp", 0)
    im = cv2.resize(im, (24, 24))
    # im.show()
    im_array = numpy.array(im)
    #print(im_array[:24])
    img_feature = NPDFeature(im_array).extract()
    print(img_feature[:300])