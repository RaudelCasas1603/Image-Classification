import os
import pickle                   # To put the data into a file

import cv2
# import matplotlib.pyplot as plt
import numpy as np

# For the progress bar
from tqdm import tqdm

from . import DEFAULT_DATA_DIR, DEFAULT_DATASET_FILE, RESIZE_WIDTH, RESIZE_HEIGHT

class DataSetCreate:
    def __init__(self, directory : str = DEFAULT_DATA_DIR, filename : str = DEFAULT_DATASET_FILE):
        self.directory = directory
        self.filename = filename
        self.data = []
        self.labels = []

                
    def save(self, savefilename : str = ""):
        if savefilename == "":
            savefilename = self.filename
        
        dataset_file = open(savefilename, "wb")
        pickle.dump(
            {
                "data" : self.data,
                "labels" : self.labels
            },
            dataset_file
        )

        dataset_file.close()

    def build(self):
        # Iterate each image from each directory
        for d in os.listdir(self.directory):
            for img_path in tqdm(os.listdir(os.path.join(self.directory, d)),
                                 desc = f"Processing images for {d}"):
                img = cv2.imread(os.path.join(self.directory, d, img_path))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Rescaled the image
                re_img = cv2.resize(img_gray, (RESIZE_WIDTH, RESIZE_HEIGHT))
                norm_img = re_img / 255.0  # Rescale the values from 0 to 1

                # Append the data
                self.data.append(np.array(norm_img))
                self.labels.append(d)
                

                    
            

        
