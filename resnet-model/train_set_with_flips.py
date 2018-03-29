import os
from PIL import Image

class GraphTrendDataset(Dataset): 
    def __init__(self, **kwargs):
        # load in the data 
        self.photos_path = kwargs['photos_path']
        self.labels_path = kwargs['labels_path']
        self.transform = kwargs['transform']
        self.flip = kwargs.get('flip', True)
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
        mylen = len(self.images)
        if self.flip: 
            mylen = 2*mylen
        return mylen

    def __getitem__(self, idx): 
        img_idx = idx//2 #int division 
        flip = idx % 2

        image = Image.open(os.path.join(self.photos_path, self.images[img_idx]))
        image = image.convert('RGB')
        label = self.labels[img_idx] 

        if flip: # the number is odd
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = -label
        
        image = self.transform(image)
        label = label + 1 # go from -1 to 1 to 0 to 2

        return (image, label)
