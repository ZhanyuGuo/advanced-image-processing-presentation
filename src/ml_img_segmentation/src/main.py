from model import MRFSegmentation, NBSegmentation, Segmentation
import cv2 as cv
import os

if __name__== '__main__':
    label_path = os.path.abspath(os.path.join(__file__, "../../assets/label.png"))
    src_path = os.path.abspath(os.path.join(__file__, "../../assets/src.jpg"))
    label_img = cv.cvtColor(cv.imread(label_path, 1), cv.COLOR_BGR2RGB)
    src_img = cv.cvtColor(cv.imread(src_path, 1), cv.COLOR_BGR2RGB)

    # ============================ Naive Bayes Segmentation ===================================
    nb_seg = NBSegmentation("Naive Bayes")
    nb_seg.train(label_img, src_img)
    nb_seg_img = nb_seg.segmentation(src_img)
    # nb_seg.show(src_img)

    # ========================== Markov Random Field Segmentation =============================
    mrf_seg = MRFSegmentation("Markov Random Field", beta=100, pre_model=nb_seg)
    mrf_seg.train(label_img, src_img)
    mrf_seg_img = mrf_seg.segmentation(src_img)
    # # mrf_seg.show(src_img)

    # 
    # print(Segmentation.iou(cv.cvtColor(cv.imread(label_path, 1), cv.COLOR_BGR2GRAY),
    #         nb_seg_img, {113:85, 75:0, 38:255}))
    
    # print(Segmentation.iou(cv.cvtColor(cv.imread(label_path, 1), cv.COLOR_BGR2GRAY),
    #     mrf_seg_img, {113:85, 75:0, 38:255}))
