from src.model import DEFAULT_MODEL_FILE, LoadModel
import sys
import cv2

from src.dataset import DEFAULT_DEVICE, RESIZE_WIDTH, RESIZE_HEIGHT
import numpy as np
from src.dataset import Collector


class Transcripter(Collector):
    def __init__(self, classes : dict, modelfile : str =  DEFAULT_MODEL_FILE,
                 stdout : int = sys.stdout, device : int = DEFAULT_DEVICE):
        
        self.model = LoadModel(modelfile)  # Load the model
        self.model.verbose = 0  # Remove the verbose
        self.device = device
        self.classes = classes
        self.stdout = stdout

    
    def transcript(self):
        self._initialize_device()
        
        prev_label = None
        while True:
            ret, frame = self.cam.read()
            H, W, _ = frame.shape

            # Rescaled the image
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            re_img = cv2.resize(img_gray, (RESIZE_WIDTH, RESIZE_HEIGHT))
            norm_img = re_img / 255.0  # Rescale the values from 0 to 1
            
            data = np.array(norm_img)  # Get the data
            # Reshape the input data to add the channel dimension
            data = np.reshape(data, (-1, RESIZE_WIDTH, RESIZE_HEIGHT, 1))
            prediction = self.model.model.predict(data, verbose = 0)
            prediction_index_label = np.argmax(prediction, axis = 1)[0]
            predicted_label = self.classes[prediction_index_label]
                        
            cv2.putText(frame, predicted_label, (int(H / 4), int(W / 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

            if prev_label != predicted_label:
                prev_label = predicted_label

                # Print the cached label
                print(predicted_label, end=", ", file = self.stdout)

            cv2.imshow('frame', frame)
            
            if cv2.waitKey(25) == ord('q'):  # To quit
                break
        self._shutdown_device()
        
        


