import numpy as np
from PIL import Image, ImageFilter, ImageMath
import os, shutil

def generate_images(size = [1024,1024], n = 100, output_dir = './sample_data', min_percent = 0.5, max_percent = 0.75):
    """Generates a sample of images of size size with random noise with a randomly sized gaussian filter cube"""
    input_dir = os.path.join(output_dir,'input')
    target_dir = os.path.join(output_dir,'target')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(input_dir)
    os.makedirs(target_dir)

    for i in range(n):
        img = generate_random_noise_image(size)
        img, img_labels = apply_random_gaussian_filter(img,min_percent,max_percent)
        print("Generating image/label pair {}".format(i))
        img.save(os.path.join(input_dir,'input_{:03d}.png'.format(i)))
        img_labels.save(os.path.join(target_dir,'target_{:03d}.png'.format(i)))


def generate_random_noise_image(size):
    x, y = size[0], size[1]
    im = np.random.randint(256, size=(x, y))
    im = Image.fromarray(np.uint8(im))
    return im

def apply_random_gaussian_filter(img, min_percent, max_percent):
    if min_percent > max_percent:
        raise ValueError('min_percent must be less than or equal to max_percent')
    im_width, im_height = img.size
    im_area = im_width * im_height
    im_area_min = int(min_percent * im_area)
    im_area_max = int(max_percent * im_area)

    rand_blur_area = np.random.randint(im_area_min, im_area_max)

    if rand_blur_area > im_width:
        rand_width = np.random.randint(im_width)+1

        rand_height = int((rand_blur_area/rand_width) - 1)
        rand_width = int(rand_width -1)
    else:
        rand_width = np.random.randint(rand_blur_area)+1
        rand_height = int( (rand_blur_area/rand_width) -1 )
        rand_width = int(rand_width - 1)

    rand_coord_left = np.random.randint(im_width)
    rand_coord_top = np.random.randint(im_height)
    rand_coord_right = rand_coord_left + rand_width
    rand_coord_bottom = rand_coord_top + rand_height

    if(rand_coord_right >= im_width):
        offset = rand_coord_right - (im_width-1)
        rand_coord_left = rand_coord_left - offset
        rand_coord_right = rand_coord_right - offset
    if(rand_coord_bottom >= im_height):
        offset = rand_coord_bottom - (im_height-1)
        rand_coord_top = rand_coord_top - offset
        rand_coord_bottom = rand_coord_bottom - offset

    crop_box = (rand_coord_left, rand_coord_top, rand_coord_right, rand_coord_bottom)

    img_crop = img.crop(crop_box)
    img_crop = img_crop.filter(ImageFilter.GaussianBlur())
    img.paste(img_crop,box=crop_box)

    #Create label image
    img_label = Image.new(mode='L', size=(im_width, im_width))
    img_label_crop = img_label.crop(box=crop_box)
    img_label_crop = ImageMath.eval('a+255', a=img_label_crop)
    img_label.paste(img_label_crop,box=crop_box)

    return (img,img_label)

if __name__ == "__main__":
    generate_images()
