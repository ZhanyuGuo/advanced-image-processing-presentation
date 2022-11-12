import numpy as np
from mmseg.core.evaluation import mean_iou

def test_mean_iou():
    pred_size = (10, 30, 30)
    num_classes = 19
    ignore_index = 255
    results = np.random.randint(0, num_classes, size=pred_size)
    label = np.random.randint(0, num_classes, size=pred_size)
    label[:, 2, 5:10] = ignore_index
    ret_metrics = mean_iou(results, label, num_classes, ignore_index)
    all_acc, acc, iou = ret_metrics["aAcc"], ret_metrics["Acc"], ret_metrics["IoU"]


if __name__ == "__main__":
    test_mean_iou()
