import Augmentor
import os

def Augmentation(path):

    list_images = [os.path.join(path, f) for f in os.listdir(path)]
    n=len(list_images)


    p = Augmentor.Pipeline(path)
    # p.rotate90(probability=0.5)
    p.rotate270(probability=0.5)
    p.flip_left_right(probability=0.6)
    # p.flip_top_bottom(probability=0.3)
    # p.crop_random(probability=1, percentage_area=0.5)
    # p.resize(probability=1.0, width=120, height=120)

    p.sample(3*n)

data_dir="./Lobular_carcinoma"


Augmentation(data_dir)

