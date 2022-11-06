from model import MRFSegmentation, NBSegmentation
import cv2 as cv
import os

if __name__== '__main__':
    label_path = os.path.abspath(os.path.join(__file__, "../../assets/label.png"))
    src_path = os.path.abspath(os.path.join(__file__, "../../assets/src.jpg"))
    src_img = cv.cvtColor(cv.imread(src_path, 1), cv.COLOR_BGR2RGB)

    # ============================ Naive Bayes Segmentation ===================================
    nb_seg = NBSegmentation("Naive Bayes")
    nb_seg.train(label_path, src_path)
    nb_seg.show(src_img)

    # ========================== Markov Random Field Segmentation =============================
    # mrf_seg = MRFSegmentation("Markov Random Field", beta=100, pre_model=nb_seg)
    # mrf_seg.train(label_path, src_path)
    # mrf_seg.show(src_img)
