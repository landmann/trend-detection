import os
import cv2
import glob 
import copy
import pyocr
import skimage
import operator
import pytesseract
import numpy as np
import scipy as scp
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from IPython.display import display
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance

class Lang():
    def __init__(self):
        self.tool = pyocr.get_available_tools()[0]
        self.lang = self.tool.get_available_languages()[0]

class LineChart():
    def __init__(self, filename, K=3, folder=None):
        self.in_rgb = True
        self.min_filter = 3
        if folder == None:
            self.folder = '/'.join(filename.split('/')[:-1])
            self.filename = filename
        else:
            self.folder = folder
            self.filename = folder+filename
        self.open_image()
        self.set_from_original()
        self.lang = Lang()
        self.output_size = (224, 224)
        self.labels = []
        
        #### assumptions
        
        # for kmeans clustering
        self.K = K
        
        # for HoughLinesP
        self.num_points_grid = 0.25 # proportion of min(heigth, width) it should span.
        self.canny_gaussian_blur = 3
        self.max_line_gap = 10 #if pixels are sparse, 10 is how much they can be separated to count as one line.
        self.min_line_length =  1/7 # lines are one seventh of the image (grids partitioned in 7)
        
        # For trend detection
        self.ocr_noise = [' ','','.',',',';','W','w']
        self.title_position = 10
        self.y_value_change_threshold = 10 #if it starts and ends 10 pixel difference, detect slope.
        self.background_threshold = 0.25 # if color is 25% of the image, it is considered part of the background.
        self.pixel_diff_threshold = 20 # if a line doesn't fluctuate more than this, it is considered a gridline
        self.min_pixel_dist = 10 # if the average distance between consecutive pixels is more than 10, it is not a line.
        self.min_grid_span = 0.8 # leftmost minus right most pixels within the same line (not changing in x or y),
                                 #     must span at least this fraction of the image.
        
        # Initializers
        self.set_foreground()
        
        # Labels
        self.labels_to_id = {
            "DECREASING": '-1',
            "INCREASING": '1',
            "NO TREND"    : '0'
        } 

        
    def open_image(self):
        """Open an PIL Image from file."""
        self.orig_image = Image.open(self.filename)
        if self.in_rgb:
            self.orig_image = self.orig_image.convert("RGB")
        if self.min_filter:
            self.orig_image.filter(ImageFilter.MinFilter(self.min_filter))
        
    def set_from_original(self):
        """Reset parameters to original image."""
        self.image = self.orig_image
        self.update_img()
        self.update_size()
    
    def update_img(self):
        """Update numpy array from Image."""
        self.img = np.array(self.image)
    
    def update_image(self):
        """Update image from numpy array."""
        self.image = Image.fromarray(self.img)
        
    def update_size(self):
        """Update size parameters from image."""
        self.size = self.image.size
        self.width, self.height = self.size
    
        
    def set_size(self, size=None):
        """Set size of the image. Defaults to self.output_size"""
        if not size:
            size = self.output_size
        self.img = cv2.resize(self.img, size)
        self.update_image()
        self.update_size()

    def display(self):
        """Show image anywhere in file. Used in middle of code to see progress."""
        display(self.image)
        
    def get_text(self):
        """Returns one string of all text detected."""
        txt = self.lang.tool.image_to_string(
            self.image,
            lang=self.lang,
            builder=pyocr.builders.TextBuilder()
        )
        return txt
    
    
    def get_line_and_word_boxes(self):
        """line.word_boxes is a list of word boxes (the individual words in the line)
           line.content is the whole text of the line
           line.position is the position of the whole line on the page (in pixels)"""
        line_and_word_boxes = self.lang.tool.image_to_string(
            self.image, lang="eng",
            builder=pyocr.builders.LineBoxBuilder()
        )
        return line_and_word_boxes


    def get_word_boxes(self):
        """Returns boxes around words separated by a space.
            box.content is the word in the box.
            box.position is the position on the page (in pixels)"""
        word_boxes = self.lang.tool.image_to_string(
            self.image,
            lang="eng",
            builder=pyocr.builders.WordBoxBuilder()
        )
        return word_boxes

    def get_digits(self):
        """Returns a string of digits."""
        digits = self.lang.tool.image_to_string(
            self.image,
            lang=lang,
            builder=pyocr.tesseract.DigitBuilder()
        )
        return digits
        
    def get_pixel(self, i, j):
        """Get the pixel from the given image"""
        # Inside image bounds?
        if i > self.width or j > self.height:
            print("Pixel out of bounds")
            return None

        # Get Pixel
        pixel = self.image.getpixel((i, j))
        return pixel
        
    # Get num_pixels and how many times they occur
    def get_colors(self, maxcolors=None):
        """If the maxcolors value is exceeded, the method stops counting and returns None. 
          The default maxcolors value is 256. To make sure you get all colors in an image, 
          you can pass in width*height """
        if maxcolors:
            return self.image.getcolors(maxcolors)
        return self.image.getcolors(self.width*self.height)
        
    def get_sorted_pixels(self):
        """Get most k used color value"""
        # (but make sure you have lots of memory before you do that on huge images).
        # [::-1] reverses order
        return sorted(self.get_colors(), key=lambda x: x[0])[::-1]
    
    def color_quantization(self):
        """Group pixels by color using k-means and color them all as the average color.
           Taken from opencv documentation: py_kmeans_opencv/py_kmeans_opencv"""
        Z = self.img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        # Criteria arguments for termination of algorithm:
        #  -type:
        #    cv2.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached. 
        #    cv2.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter. 
        #    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER - 
        #                stop the iteration when any of the above condition is met.
        #  -max_iter: (10 recommended)
        #  -epsilon: 
        #     accuracy ; cluster average moved 'eps' from last location (1.0 recommended)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Arguments into kmeans are:
        #   -datapoints
        #   -nclusters
        #   -criteria
        #   -attempts: num times it will use different initial labels ; (10 recommended)
        #   -flags: how initial centers are taken
        
        ret,label,center= cv2.kmeans(Z,self.K,None, criteria ,10 , cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((self.img.shape))
        
        self.img = res2
        self.update_image()
        
    def crop_title(self):
        """Find the first characters in the image and crop them out."""
        top_line_and_words = self.get_line_and_word_boxes()
        top_line_and_words = [boxes for boxes in self.get_line_and_word_boxes() if boxes.content not in self.ocr_noise]
        if top_line_and_words:
            topleft, bottomright = top_line_and_words[0].position
            print(topleft, bottomright)
            if topleft[1] <= self.title_position:
                self.image = self.image.crop((0, bottomright[1], self.width, self.height))
                self.update_img()
                self.update_size()
    
    def crop_to_gridline(self):
        """Crops image to contain only what's inside the actual graph.
           Taken from opencv documentation: py_houghlines/py_houghlines"""
        
        tmp_image = self.image.copy()
        tmp_image = ImageEnhance.Sharpness(tmp_image).enhance(25)
        tmp_image = ImageEnhance.Color(tmp_image).enhance(25)
        tmp_img = np.array(tmp_image)
        
        gray = cv2.cvtColor(tmp_img,cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection works as follows:
        #  1. Noise reduction using 5x5 gaussian filter
        #  2. Finding intensity gradients in an image using Sobel kernel (horiz, and vert)
        #  3. Non-maximum Suppression using local maximums.
        #  4. Thresholding: minVal to discard any edges, maxVal for 'sure edge'.
        #    if line between minVal and maxVal, line has to go into maxVal at some point to be considered an edge.
        
        # # minVal=50, maxVal=150, and gaussianFilter 5 are standard.
        edges = cv2.Canny(gray, 50, 150, apertureSize = self.canny_gaussian_blur) #Canny edge detection
        xmin, ymin = self.width, self.height
        xmax = ymax = 0
        
        # HoughLinesP finds line segments in a binary image using probabilistic Hough transforms.
        #   - 8-bit image
        #   - Rho: Pixels two lines can differ by to be considered part of the same line (1 pixel recommended)
        #   - theta: The angle two lines can differ by to be considered part of the same line (1 degree recommended)
        #   - number of points in a line
        #   - minLineLength - minimum line length in pixels
        #   - maxLineGap - max allowed gap between points on the same line to link them
        lines = cv2.HoughLinesP(edges,
                                1 ,
                                np.pi/180, 
                                int(min(self.size)*self.num_points_grid),
                                int(min(self.size)*self.min_line_length), 
                                self.max_line_gap)
        
        if type(lines)!=np.ndarray:
            print("Could not find any lines.")
            return
        
        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs(x1-x2)>self.min_line_length or abs(y1-y2)>self.min_line_length:
                    xmin = min(xmin, x1, x2)
                    xmax = max(xmax, x1, x2)
                    ymin = min(ymin, y1, y2)
                    ymax = max(ymax, y1, y2)

        if ymax==ymin or xmax==xmin:
            return
        self.image = self.image.crop((xmin, ymin, xmax, ymax))
        self.update_img()
        self.update_size()
        
        
    def separate_colors(self):
        """Separates colors. Returns list of images and pixel locations."""
        colors = self.get_sorted_pixels()
        colors_dict = dict((val[1], Image.new('RGB', self.size, (255,255,255))) 
                            for val in colors)
        pixel_dict = dict((img, []) for img in colors_dict.keys())

        pix = self.image.load()
        for i in range(self.width):
            for j in range(self.height):
                if pix[i,j] in colors_dict:
                    colors_dict[pix[i,j]].putpixel((i,j),(0,0,0))
                    pixel_dict[pix[i,j]].append((i, j))

        return [(color, colors_dict[color], pixels) for color, pixels in pixel_dict.items()] 
    
    def set_foreground(self):
        """Extracts foreground based on dominant color."""
        
        # crop, resize, color quantize
        self.crop_to_gridline()
        self.crop_title()
        self.set_size()
        self.color_quantization()
        self.display()
        img_and_pix = self.separate_colors()
        colors, images, pixels = zip(*img_and_pix)
        self.foreground = images[0]
        self.foreground = self.foreground.convert('1')
        
    def separate_one_line(self):
        self.labels = []
        """Separates graphs when they have one line. Uses foreground image only."""
        bmarray = np.array(self.foreground)
        # Find all foreground pixels
        allx, ally = np.where(bmarray)
        
        # Find rows or columns spanning entire picture and blend them with background.
        yfills = np.where(np.bincount(ally) > (self.height*self.min_grid_span))[0]
        xfills = np.where(np.bincount(allx) > (self.width*self.min_grid_span))[0]
        bmarray[:,yfills] = np.array(bmarray[:,yfills]*0, dtype=bool)
        bmarray[xfills,:] = np.array(bmarray[xfills,:]*0, dtype=bool)
        
        # make dataframe 
        ally, allx = np.where(bmarray)
        bm1 = list(zip(allx, ally))
        df = pd.DataFrame(bm1, columns=['x', 'y'])
        df = df.sort_values(['x', 'y'])
        df.reset_index(inplace=True, drop=True)
        
        separation = 10
        lines = [{'last':[], 'pixels':[], 'id':0}]

        for x in df['x'].unique():
            prev_pix = -2*separation
            groups = []
            group_num = -1
            for y in df[df['x']==x]['y'].values:
                if y - prev_pix <= separation:
                    groups[group_num].append((y))
                else:
                    groups.append([(y)])
                    group_num += 1 
                prev_pix = y

            new_lines = []
            for group in groups:
                done = False
                values = [(x, y) for y in group]
                for line in lines:
                    for last_x, last_y in line['last']:
                        if np.any(abs(last_y-((np.array(group)))) <= separation) and abs(last_x - x)<=separation:
                            line['pixels'].extend(values)
                            line['last'] = values
                            done = True
                            break
                    if done:
                        break
                else:
                    new_lines.append(values)

            for newline in new_lines:
                lines.append({'last': newline, 'pixels':newline, 'id':len(lines)})
                
        ixs = sorted(lines, key=lambda x: len(x['pixels']), reverse=True)
        num_pixels = [x['id'] for x in ixs]

        images = dict((line['id'], Image.new('RGB', self.size, (0,0,0))) 
                            for line in lines)

        for line in lines:
            for pixel in line['pixels']:
                images[line['id']].putpixel(pixel,(255,255,255))
                
        line = images[num_pixels[0]]


        img = np.array(line.convert('1'))
        if np.any(img):
            allx, invy = np.where(img)[:2]
            ally = img.shape[0] - invy
            pix = list(zip(allx, ally))

            df = pd.DataFrame(pix).groupby(0).agg(np.mean)
            xvals = df.index
            yvals = df.values.flatten().astype(int)
            slope, intercept, rvalue, pvalue, stderr = scp.stats.linregress(xvals, yvals)

            change = slope*len(yvals)
            if pvalue > 0.05 or abs(change) < 0.01:
                self.add_label("NO TREND")
            else:
                if slope < 0:
                    self.add_label("DECREASING")
                else:
                    self.add_label("INCREASING") 
        else:
            self.add_label("NO TREND")
        return self.labels
        
        
    

    def get_all_trends(self, verbose=False):
        """Parse through the images and separate the graphs that are lines."""
        self.labels = []
        inferred_lines = []
        
        # crop, resize, color quantize
        self.crop_to_gridline()
        self.crop_title()
        self.set_size()
        self.color_quantization()
        
        if verbose:
            print("Original Image")
            display(self.orig_image)
            print("Cropped and color quantized:")
            self.display()
            
        # separate colors into color pixel, images, and pixels that belong to it
        img_and_pix = self.separate_colors()
        colors, images, pixels = zip(*img_and_pix)
        
        if verbose:
            print("LEN IMGS", len(images), "; SHOULD BE", len(self.separate_colors()))
    
        for i, image in enumerate(images):
            
            # separate into x and y pixels
            pix = pixels[i]
            inferred_lines.append(img_and_pix[i])
            x,y = zip(*np.array(pix))
            
            if verbose:
                print('len(set(y))>20', len(set(y)))
                print('len(set(x))>20', len(set(x)))
                print('len(pix)<= {}; actual: {}'.format(self.background_threshold*np.prod(self.size), len(pix)))
                display(image)
                
            # If pixels don't fluctuate more than pixel_diff_threshold, 
            #   or size of color is greater than background_threshold..
            if len(pix) <= self.background_threshold*np.prod(self.size):
                if len(set(y))>self.pixel_diff_threshold \
                    and len(set(x))>self.pixel_diff_threshold:
                
                    # take difference between pixels
                    d = np.diff(pix, axis=0)
                    segdists = np.sqrt((d ** 2).sum(axis=1))

                    # Check if one line alone in the bitmap spans min_grid_length.
                    # np.argmax(np.bincount(x)) -> keep x constant
                    pot_vert_line = [j for i, j in pix if i == np.argmax(np.bincount(x))]
                    pot_hor_line = [i for i, j in pix if j == np.argmax(np.bincount(y))]
                    
                    if verbose:
                        print("Passed 0.25 pixel threshold and straight line threshold")
                        print("sum(segdists)/len(segdists)<7", sum(segdists)/len(segdists))
                        print("Y LINE ", pot_vert_line[0], pot_vert_line[-1], self.height*self.min_grid_span)
                        print("X LINE" , pot_hor_line[0], pot_hor_line[-1], self.width*self.min_grid_span)

                    if sum(segdists)/len(segdists)<self.min_pixel_dist and \
                        len(pot_vert_line)<(self.height*self.min_grid_span) and \
                        len(pot_hor_line)<(self.width*self.min_grid_span):


#                         inferred_lines.append(img_and_pix[i])
                        # display(image)

                        actual_y = self.height - np.array(pix)[:,1]
                        
                        df = pd.DataFrame(pix).groupby(0).agg(np.mean)
                        xvals = df.index
                        yvals = df.values.flatten().astype(int)
                        slope, intercept, rvalue, pvalue, stderr = scp.stats.linregress(xvals, yvals)
                        
                        change = slope*len(yvals)
                        if pvalue > 0.05 or abs(change) < 0.01:
                            self.add_label("NO TREND")
                        else:
                            if slope < 0:
                                self.add_label("DECREASING")
                            else:
                                self.add_label("INCREASING")       

            if verbose:                    
                print("-"*50)
        if not self.labels:
            self.add_label("NO TREND")
        return inferred_lines, self.labels
        
    def add_label(self, trend_tag):
#         print("Trend detected: {}".format(trend_tag))
        self.labels.append(self.labels_to_id[trend_tag])
