import os
import hydra
from PIL import Image


def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)


def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)

    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)

        if (i + 1) % 100 == 0:
            print("[{}/{}] Resized the images and saved in to '{}'.".format(i + 1, num_images, output_dir))


@hydra.main(config_path="config/config.yaml")
def main(cfg):
    image_dir = cfg.resize.image_dir
    output_dir = cfg.resize.output_dir
    image_size = [cfg.resize.image_size, cfg.resize.image_size]

    resize_images(image_dir, output_dir, image_size)