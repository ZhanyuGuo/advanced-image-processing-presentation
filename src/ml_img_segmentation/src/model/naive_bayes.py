import numpy as np
from .segmentation import Segmentation

class NBSegmentation(Segmentation):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    '''
    * @breif: Image segmentation based on Naive Bayes model.
    * @param[in]: img   ->  source image(RGB or Gray)
    * @retval: segmentation image
    '''
    def segmentation(self, img: np.ndarray) -> np.ndarray:
        if not self.trained_:
            raise InterruptedError("Please call `train` function first!")

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
                            cls_posterior *= Segmentation.normalPdf(val, label[c][1], label[c][2])
                    else:
                        val = img[i][j]
                        cls_posterior *= Segmentation.normalPdf(val, label[0][1], label[0][2])

                    # Calculate class posterior
                    if (cls_posterior > max_probabilty):
                        max_probabilty, best_class = cls_posterior, index
                predict_array[i][j] = self.class_color[best_class]
                
        return predict_array
