# Some global variables
DEFAULT_DATA_DIR = "./data"
DEFAULT_DATASET_FILE = "dataset.pickle"
DEFAULT_AMOUNT_OF_SIGNS = 26    # letters
DATA_SIZE = 1000             # 1000 frames
DEFAULT_DEVICE = 0

RESIZE_WIDTH = 200
# 480

RESIZE_HEIGHT = 200
# 640

from .collect import Collector
from .create import DataSetCreate
