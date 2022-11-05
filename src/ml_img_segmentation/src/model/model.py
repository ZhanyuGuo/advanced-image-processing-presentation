import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import random

class Segmentation:
    def __init__(self, path: str, **kwargs) -> None:
        # read image
        self.src = cv.imread(os.path.abspath(os.path.join(__file__, path)), 1)
        self.rgb = cv.cvtColor(self.src, cv.COLOR_BGR2RGB)
        self.gray = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)

        # algorithm parameters
        self.alpha = kwargs.pop('alpha', 10)
        self.beta = kwargs.pop('beta', 10)

        # initialize labels information
        self.class_info = []
        for label, pro in kwargs.items():
            tmp_src = cv.imread(os.path.abspath(os.path.join(__file__, "../../../assets/" + label + '.jpg')))
            # gray, RGB
            self.class_info.append([[pro, np.mean(tmp_src), np.var(tmp_src)],
                  [pro, np.mean(tmp_src[:, :, 2]), np.var(tmp_src[:, :, 2])],
                  [pro, np.mean(tmp_src[:, :, 1]), np.var(tmp_src[:, :, 1])],
                  [pro, np.mean(tmp_src[:, :, 0]), np.var(tmp_src[:, :, 0])]])

        # configure class color
        self.class_color = np.arange(0, 256, int(255 / (len(self.class_info) - 1)))

    '''
    * @breif: Image segmentation based on Naive Bayes model.
    * @param[in]: img   ->  source image(RGB or Gray)
    * @retval: segmentation image
    '''
    def nbSegmentation(self, img: np.ndarray) -> np.ndarray:
        # get image's shape
        channels = 0
        try:    rows, cols, channels = img.shape
        except: rows, cols = img.shape
        predict_array = np.zeros([rows, cols], dtype=float)

        # travel all pixels
        for i in range(0, rows):
            for j in range(0, cols): 
                max_probabilty, best_class = 0, -1

                # Maximum likelihood
                for index, label in enumerate(self.class_info):
                    # Prior probability
                    cls_posterior = label[0][0]
                    if channels:
                        for c in range(channels):
                            val = img[i][j][c]
                            cls_posterior *= Segmentation.normalPdf(val, label[c + 1][1], label[c + 1][2])
                    else:
                        val = img[i][j]
                        cls_posterior *= Segmentation.normalPdf(val, label[0][1], label[0][2])

                    # Calculate class posterior
                    if (cls_posterior > max_probabilty):
                        max_probabilty, best_class = cls_posterior, index
                predict_array[i][j] = self.class_color[best_class]
                
        return predict_array

    '''
    * @breif: Image segmentation based on Conditional Random Field(simulated annealing).
    * @param[in]: img   ->  source image(RGB or Gray)
    * @retval: segmentation image
    '''
    def crfSegmentation(self, img: np.ndarray, w: np.ndarray, init_temp=800) -> np.ndarray:
        # get image shape
        rows, cols = w.shape
        predict_array = np.zeros([rows, cols], dtype=float)

        # initialize energe and temperature
        energy = 0.0
        for i in range(rows):
            for j in range(cols):
                energy += self.__calEnergy(w[i][j], (i, j), img, w)
        current_temp = init_temp

        iteration =0
        while (iteration < 500):
            # try new label
            x, y = random.randint(0, rows - 1), random.randint(0, cols - 1)
            labels = [i for i in range(len(self.class_info))]
            labels.remove(w[x][y])
            new_label = labels[random.randint(0, len(labels) - 1)]

            # delta energy between old label and new label
            delta = self.__calEnergy(new_label, (x, y), img, w) - self.__calEnergy(w[x][y], (x, y), img, w)
            if (delta <= 0):
                w[i][j] = new_label
                energy += delta
            else:
                p = -delta / current_temp
                if random.uniform(0, 1) < p:
                    w[i][j] = new_label
                    current_energy += delta
            current_temp *= 0.95
            iteration += 1

        for i in range (0, rows):
            for j in range(0, cols):
                predict_array[i][j] = self.class_color[w[i][j]]

        return predict_array

    '''
    * @breif: Visulize the segement image
    * @param[in]: img   ->  source image(RGB or Gray)
    * @retval: None
    '''
    def show(self, img: np.ndarray) -> None:
        # get image shape
        rows, cols = img.shape[0], img.shape[1]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('Comparision', fontsize=20)
        
        # Initial Image
        ax1.set_title("Initial Image")
        ax1.imshow(img, cmap='gray')

        # Naive Bayes Segmentation Image
        bayes_seg_img = self.nbSegmentation(img)
        ax2.set_title('Naive Bayes Segmentation Image')
        ax2.imshow(bayes_seg_img, cmap='gray')

        # Conditional Random Field Image
        w = np.zeros([rows, cols], dtype=int)
        color_index = {color: label for label, color in enumerate(self.class_color)}
        for i in range(0, rows):
            for j in range(0, cols):
                w[i][j] = color_index[bayes_seg_img[i][j]]
        crf_seg_img = self.crfSegmentation(img, w)
        ax3.set_title('Conditional Random Field Image')
        ax3.imshow(crf_seg_img, cmap='gray')
        
        plt.show()

    '''
    * @breif: Visulize the segement image
    * @param[in]: label ->  pixel's label in location `index`
    * @param[in]: index ->  target pixel's location, (x, y)
    * @param[in]: img   ->  source image(RGB or Gray)
    * @param[in]: w     ->  current labels field
    * @retval: energy
    '''
    def __calEnergy(self, label: int, index: tuple, img: np.ndarray, w: np.ndarray) -> float:
        # get image's shape
        channels = 0
        try:    rows, cols, channels = img.shape
        except: rows, cols = img.shape

        i, j = index
        energy = 0.0
        
        if channels:
            for c in range(channels):
                val = img[i][j][c]
                mean, var = self.class_info[label][c + 1][1], self.class_info[label][c + 1][2]
                energy += np.log(np.sqrt(2 * np.pi * var)) + ((val - mean)**2) / (2 * var)
        else:
            val = img[i][j]
            mean, var = self.class_info[label][0][1], self.class_info[label][0][2]
            energy += np.log(np.sqrt(2 * np.pi * var)) + ((val - mean)**2) / (2 * var)

        # clique energy(Potts model)
        neighbor = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for dx, dy in neighbor:
            if 0 <= i + dx < rows and 0 <= j + dy < cols:
                energy = energy + self.beta if label == w[i + dx][j + dy] else energy

        return energy 

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