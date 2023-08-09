import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

aug_seq = sometimes(iaa.Sequential([
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.9, 1.1)),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(scale={"x": (0.9, 1.0), "y": (0.9, 1.0)},rotate=(-180, 180),)
    ], random_order=True)) # apply augmenters in random order

def aug_sequential(image_data:Image.Image):
    """
    调用imgaug进行图像增强，由于训练图像为PIL.Image.Image格式，存在一定转换
    """
    image_data = np.array(image_data)
    aug_data = aug_seq(image=image_data)
    aug_data = Image.fromarray(aug_data)
    return aug_data

if __name__ == "__main__":
    image_path = r"D:\py_project\seal\faker_seal_image\0.png"
    image_data = Image.open(image_path)
    for i in range(50):
        images_aug = aug_sequential(image_data)
        images_aug.show()