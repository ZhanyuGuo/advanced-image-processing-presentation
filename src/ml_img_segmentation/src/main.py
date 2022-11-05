from model import Segmentation

if __name__== '__main__':
    seg = Segmentation("../../../assets/test_mini.jpg", sky=0.30, road=0.20, grass=0.50)
    seg.show(seg.rgb)