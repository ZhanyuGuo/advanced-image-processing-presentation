import os
import cv2
import json
import numpy as np


def cityscapes_classes():
    """Cityscapes class names for external use."""
    return [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]


def main():
    file_path = os.path.dirname(__file__)
    json_path = os.path.abspath(
        os.path.join(file_path, "../demo/lindau_000058_000019_gtFine_polygons.json")
    )
    save_path = os.path.abspath(
        os.path.join(file_path, "../demo/lindau_000058_000019_label_resize.png")
    )

    data = json.load(open(json_path))
    d_ratio = 2

    h = int(data["imgHeight"] / d_ratio)
    w = int(data["imgWidth"] / d_ratio)
    # print(h, w)

    img = np.zeros((h, w))
    classes = cityscapes_classes()
    # print(classes.index("road"))
    for i in range(len(data["objects"])):
        name = data["objects"][i]["label"]
        points = data["objects"][i]["polygon"]
        for i in range(len(points)):
            points[i][0] /= d_ratio
            points[i][1] /= d_ratio

        if name in classes:
            color = classes.index(name)
        else:
            color = 255
        img = cv2.fillPoly(img, [np.array(points, dtype=int)], color)

    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    main()
