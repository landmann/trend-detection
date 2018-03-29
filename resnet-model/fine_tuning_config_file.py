# Learning rate parameters
BASE_LR = 0.0001
EPOCH_DECAY = 100 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1 # factor by which the learning rate is reduced.


# DATASET INFO
NUM_CLASSES = 3 # set the number of classes in your dataset
DATA_PATH = '~/graph-trend-understanding/data/'
TEST_DATA_PATH = '~/data/images/test/'


# DATALOADER PROPERTIES
BATCH_SIZE = 75 # Set as high as possible. If you keep it too high, you'll get an out of memory error.

### GPU SETTINGS
CUDA_DEVICE = 0 # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 1 # set to 1 if want to run on gpu.


# SETTINGS FOR DISPLAYING ON TENSORBOARD
USE_TENSORBOARD = 0 #if you want to use tensorboard set this to 1.
TENSORBOARD_SERVER = "YOUR TENSORBOARD SERVER ADDRESS HERE" # If you set.
EXP_NAME = "fine_tuning_experiment" # if using tensorboard, enter name of experiment you want it to be displayed as.
