from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

class Segmentation(ABC):
    def __init__(self, name: str) -> None:
        # class information with RGB channels
        self.class_info = []
        # one color for each category
        self.class_color = None
        # whether trained or not, run model only after training
        self.trained_ = False
        # model's name
        self.name = name


    def train(self, label_path: str, src_path: str):
        # source image
        src = cv.imread(src_path, 1)
        rgb = cv.cvtColor(src, cv.COLOR_BGR2RGB)

        # count pixel color
        with Image.open(label_path) as img:
            width, height = img.size
            labeled = img.convert('RGB')
            color_count = {}
            for x in range(width):
                for y in range(height):
                    pixel = labeled.getpixel((x, y))
                    if pixel in color_count:
                        color_count[pixel] = np.dstack((color_count[pixel], rgb[y, x, :]))
                    else:
                        color_count[pixel] = rgb[y, x, :]

        # initialize labels information
        for _, val in color_count.items():
            pro = val.shape[2] / (width * height)
            self.class_info.append([
                        [pro, np.mean(val[:, 0, :]), np.var(val[:, 0, :])],
                        [pro, np.mean(val[:, 1, :]), np.var(val[:, 1, :])],
                        [pro, np.mean(val[:, 2, :]), np.var(val[:, 2, :])]
                    ]) 
        
        # configure class color
        self.class_color = np.arange(0, 256, int(255 / (len(self.class_info) - 1)))

        # trained finished
        self.trained_ = True


    @abstractmethod
    def segmentation(self, img: np.ndarray):
        pass

    '''
    * @breif: Visulize the segement image
    * @param[in]: img   ->  source image(RGB or Gray)
    * @retval: None
    '''
    def show(self, img: np.ndarray) -> None:
        if not self.trained_:
            raise InterruptedError("Please call `train` function first!")

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Comparision', fontsize=20)
        
        # Initial Image
        ax1.set_title("Initial Image")
        ax1.imshow(img, cmap='gray')

        # Segmentation Image
        seg_img = self.segmentation(img)
        ax2.set_title(self.name + " Image")
        ax2.imshow(seg_img, cmap='gray')
        
        plt.show()

    '''
    * @breif: Probability density function of normal distribution.
    * @param[in]: x     ->  input
    * @param[in]: mean  ->  mean value
    * @param[in]: var   ->  variance
    * @retval: probability density
    '''    
    @staticmethod
    def normalPdf(x: float, mean: float, var: float) -> float:
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean)**2) / (2 * var))

    @staticmethod
    def fpr():
        pass

    @staticmethod
    def iou():
        pass
