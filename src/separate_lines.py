import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
import copy
import sys
import os

# Open an Image
def open_image(path, in_rgb=True):
    newImage = Image.open(path)
    if in_rgb:
        return newImage.convert("RGB")
    return newImage

# Save Image
def save_image(image, path):
    image.save(path, 'png')

# Create a new image with the given size
def create_image(i, j):
    image = Image.new("RGB", (i, j), "white")
    return image

# Get the pixel from the given image
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
        return None

    # Get Pixel
    pixel = image.getpixel((i, j))
    return pixel

# create numpy array from image
def make_img(image):
    return np.array(image)

# create image from numpy array
def make_image(img):
    return Image.fromarray(img)

# get most used color value
def get_topk_pixels(image, k=5):
#     If the maxcolors value is exceeded, the method stops counting and returns None. 
#     The default maxcolors value is 256. To make sure you get all colors in an image, 
#     you can pass in size[0]*size[1] 
#     (but make sure you have lots of memory before you do that on huge images).
#     [::-1] reverses order
    return sorted(image.getcolors(np.prod(image.size)), key=lambda x: x[0])[-k:][::-1]

# Separates colors. Returns list of images and pixel locations.
def separate_colors(image, k=5):
    colors = get_topk_pixels(image, k)
    colors_dict = dict((val[1], Image.new('RGB', image.size, (255,255,255))) 
                        for val in colors)
    pixel_dict = dict((img, []) for img in colors_dict.keys())
    
    pix = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            if pix[i,j] in colors_dict:
                colors_dict[pix[i,j]].putpixel((i,j),(0,0,0))
                pixel_dict[pix[i,j]].append((i, j))

    return [(color, colors_dict[color], pixels) for color, pixels in pixel_dict.items()]

def separate_lines(images):
    background = images[0]

def save_graphs(img_path):
    directory = img_path[:-4]
    if not os.path.exists(directory):
        os.makedirst(directory)
    for i, image in enumerate(images[2:]):
       image.save("{}/trend{}.png".format(directory, i)) 
    return

def save_images(filename):
    orig_image = open_image(filename)
    img_and_pix = separate_colors(orig_image, 6)
    colors = [i[0] for i in img_and_pix]
    images = [i[1] for i in img_and_pix]
    pixels = [i[2] for i in img_and_pix]
   
    foldername = filename[:-4]+'_trend/'
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    for i, img in enumerate(images[2:]):
        imgfile = "{}trend{}.png".format(foldername, i)
        print(imgfile)
        img.save(imgfile)

if __name__ == "__main__":
    folder = '../Graphs/'
    if len(sys.argv) > 0:
        path = sys.argv[1]
    else:
        # path = 'flowers-national-parks-4.png'
        path = 'LinearTrends.png'
     
    save_images(folder+path)




















