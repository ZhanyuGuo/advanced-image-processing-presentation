import numpy as np
import random
from .segmentation import Segmentation

class MRFSegmentation(Segmentation):
    def __init__(self, name: str, beta: float, pre_model: Segmentation, temp=800) -> None:
        super().__init__(name)
        # algorithm parameters
        self.beta_ = beta
        self.pre_model_ = pre_model
        self.temp_ = temp

    '''
    * @breif: Image segmentation based on Conditional Random Field(simulated annealing).
    * @param[in]: img   ->  source image(RGB or Gray)
    * @retval: segmentation image
    '''
    def segmentation(self, img: np.ndarray) -> np.ndarray:
        if not self.trained_:
            raise InterruptedError("Please call `train` function first!")

        # get image shape
        rows, cols = img.shape[0], img.shape[1]
        predict_array = np.zeros([rows, cols], dtype=int)

        # pre-labeled
        pre_seg = self.pre_model_.segmentation(img)
        w = np.zeros([rows, cols], dtype=int)
        color_index = {color: label for label, color in enumerate(self.class_color)}
        for i in range(0, rows):
            for j in range(0, cols):
                w[i][j] = color_index[pre_seg[i][j]]

        # initialize energe and temperature
        energy = 0.0
        for i in range(rows):
            for j in range(cols):
                energy += self.__calEnergy(w[i][j], (i, j), img, w)
        current_temp = self.temp_

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
    * @breif: Calculate energy
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
                mean, var = self.class_info[label][c][1], self.class_info[label][c][2]
                energy += np.log(np.sqrt(2 * np.pi * var)) + ((val - mean)**2) / (2 * var)
        else:
            val = img[i][j]
            mean, var = self.class_info[label][0][1], self.class_info[label][0][2]
            energy += np.log(np.sqrt(2 * np.pi * var)) + ((val - mean)**2) / (2 * var)

        # clique energy(Potts model)
        neighbor = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        for dx, dy in neighbor:
            if 0 <= i + dx < rows and 0 <= j + dy < cols:
                energy = energy + self.beta_ if label == w[i + dx][j + dy] else energy

        return energy 
