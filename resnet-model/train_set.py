from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image

class GraphTrendDataset(Dataset): 
    def __init__(self, **kwargs):
        # load in the data 
        self.photos_path = kwargs['photos_path']
        self.labels_path = kwargs['labels_path']
        #self.labels_path = os.path.expanduser('~/graph-trend-understanding/data/val_labels_linear.txt')
        self.transform = kwargs['transform']
        self.flip = kwargs.get('flip', False)
        self.images = []
        self.labels = []

        # read the text file
        with open(self.labels_path, 'r') as f: 
            for line in f:
                img_name, label = line.strip().split(" ")
                self.images.append(self.photos_path + img_name)
                self.labels.append(label)

        self.images = np.array(self.images, np.object)
        self.labels = np.array(self.labels, np.int64)
        print("# images found at path '%s': %d" % (self.labels_path, self.images.shape[0]))

    def __len__(self): 
        #return len(self.images)
        # going to use only 50k images
        return min(len(self.images), 50000)

    def __getitem__(self, idx): 
        image = Image.open(os.path.join(self.photos_path, self.images[idx]))
        image = image.convert('RGB')
        image = self.transform(image)
        # label is the index of the correct category
        label = self.labels[idx] 
        if self.flip: 
            label = -label
        label = label + 1 # go from -1 to 1 to 0 to 2
        return (image, label)
