from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


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

    """
    * @breif: Train the segmentation parameters
    * @param[in]: label_img ->  labeled image(RGB)
    * @param[in]: src_img   ->  source image(RGB)
    * @retval: None
    """

    def train(self, label_img: np.ndarray, src_img: np.ndarray):
        # get image's size
        rows, cols = label_img.shape[0], label_img.shape[1]

        # count pixel color
        color_count = {}
        for x in range(rows):
            for y in range(cols):
                pixel = (label_img[x, y, 0], label_img[x, y, 1], label_img[x, y, 2])
                if pixel in color_count:
                    color_count[pixel] = np.dstack(
                        (color_count[pixel], src_img[x, y, :])
                    )
                else:
                    color_count[pixel] = src_img[x, y, :]
                    # color_count[pixel] = np.dstack((src_img[x, y, :], src_img[x, y, :]))

        # initialize labels information
        for _, val in color_count.items():
            pro = val.shape[2] / (rows * cols)  # BUG: IndexError: tuple index out of range, (3, )[2] is wrong.
            self.class_info.append(
                [
                    [pro, np.mean(val[:, 0, :]), np.var(val[:, 0, :])],
                    [pro, np.mean(val[:, 1, :]), np.var(val[:, 1, :])],
                    [pro, np.mean(val[:, 2, :]), np.var(val[:, 2, :])],
                ]
            )

        # configure class color
        self.class_color = np.arange(0, 256, int(255 / (len(self.class_info) - 1)))

        # trained finished
        self.trained_ = True

    @abstractmethod
    def segmentation(self, img: np.ndarray):
        pass

    """
    * @breif: Visulize the segement image
    * @param[in]: img   ->  source image(RGB or Gray)
    * @retval: None
    """

    def show(self, img: np.ndarray) -> None:
        if not self.trained_:
            raise InterruptedError("Please call `train` function first!")

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Comparision", fontsize=20)

        # Initial Image
        ax1.set_title("Initial Image")
        ax1.imshow(img, cmap="gray")

        # Segmentation Image
        seg_img = self.segmentation(img)
        ax2.set_title(self.name + " Image")
        ax2.imshow(seg_img, cmap="gray")

        plt.show()

    """
    * @breif: Probability density function of normal distribution.
    * @param[in]: x     ->  input
    * @param[in]: mean  ->  mean value
    * @param[in]: var   ->  variance
    * @retval: probability density
    """

    @staticmethod
    def normalPdf(x: float, mean: float, var: float) -> float:
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    @staticmethod
    def iou(label_img: np.ndarray, seg_img: np.ndarray, map_dict: dict) -> float:
        # get image's size
        rows, cols = label_img.shape[0], label_img.shape[1]

        # count right prediction
        pre_count = {}
        for label, pre in map_dict.items():
            pre_count[str(label)] = 0
            pre_count[str(pre)] = 0
            pre_count[str(label) + str(pre)] = 0

        for x in range(rows):
            for y in range(cols):
                try:
                    pre_count[str(label_img[x][y])] += 1
                    pre_count[str(seg_img[x][y])] += 1
                    if map_dict[label_img[x][y]] == seg_img[x][y]:
                        pre_count[str(label_img[x][y]) + str(seg_img[x][y])] += 1
                except:
                    continue

        res = [
            pre_count[str(label) + str(pre)]
            / (
                pre_count[str(label)]
                + pre_count[str(pre)]
                - pre_count[str(label) + str(pre)]
            )
            for label, pre in map_dict.items()
        ]

        return sum(res) / len(res)
