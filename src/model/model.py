import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Include the nerual net
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model as KModel

import numpy as np
from . import DEFAULT_MODEL_FILE, DEFAULT_DATASET_FILE, IMG_WIDTH, IMG_HEIGHT
import pdb                      # For debuggin

# Maps each label to a single number with this we can perfeclty map outputs
def build_map_label(lst):
    dic = {}
    index_number = 0
    for item in lst:
        if item not in dic:
            dic[item] = index_number
            index_number += 1
    return dic

class Model:
    def __init__(self, dataset : str = DEFAULT_DATASET_FILE, modelfile = DEFAULT_MODEL_FILE):
        self.data_dict = pickle.load(open(dataset, 'rb'))
        self.modelfile = modelfile
        
        # pdb.set_trace()
        # Run the protected methods
        self._prepare_data()
        self._prepare_model()
        self._prepare_train_test_arrays()


    def _prepare_model(self):
        # Create the CNN model
        self.model = models.Sequential()
        
        self.model.add(layers.Conv2D(IMG_HEIGHT, (3, 3), activation = "relu",
                                     input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)))  # Input layer
        self.model.add(layers.MaxPool2D((2, 2)))
        self.model.add(layers.Conv2D(IMG_HEIGHT, 3, activation = "relu"))
        self.model.add(layers.MaxPool2D((2, 2)))
        
        # Then classify
        self.model.add(layers.Flatten())  # squizzy to one dimension
        self.model.add(layers.Dense(self.class_output_neurons * 20, activation = 'relu'))
        self.model.add(layers.Dense(self.class_output_neurons, activation = "softmax"))  # Output layer

    def _prepare_train_test_arrays(self):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data, self.data_output, shuffle = True, test_size = 0.3)

    def _prepare_data(self):
        # Get the amount of neuros that should have the output layer
        labels = np.asarray(self.data_dict['labels'])
        self.labels_map = build_map_label(labels)
        self.class_output_neurons = len(self.labels_map)

        # Build the output data
        labels_output = []
        for label in labels:
            labels_output.append(self.labels_map[label])

        # Build the input and output array
        self.data = np.asarray(self.data_dict['data'])
        self.data_output = np.asarray(labels_output)


    def train(self):
        # pdb.set_trace()
        # Optimze with adam and loss sparce categorlical crossentropy
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()    # Print the model summary

        self.history = self.model.fit(self.x_train, self.y_train, epochs = 10, batch_size = 32)

    def test(self):
        # pdb.set_trace()
        y_predict = np.argmax(self.model.predict(self.x_test), axis = 1)
        # The model should return [1.0, 0.0, ..., 0.0]  depending on the label and get the index
        score = accuracy_score(y_predict, self.y_test)
        
        print('{}% of samples were classified correctly !'.format(score * 100))


    def save(self):
        # Dump the model
        f = open(self.modelfile, "wb")
        pickle.dump({'model': self.model}, f)
        f.close()


# LoadModel: To load the model from a file and replace part of its brain
class LoadModel(Model):
    def __init__(self, dataset : str = DEFAULT_DATASET_FILE, modelfile = DEFAULT_MODEL_FILE):
        self.modelfile = modelfile
        self.model_dict = pickle.load(open(modelfile, "rb"))
        self.data_dict = pickle.load(open(dataset, 'rb'))
        
        # Load the model
        self.model = self.model_dict["model"]

        # Prepare the data
        self._prepare_data()
        self._prepare_train_test_arrays()
        self._shortens_train_test_arrays()  # Load the trainable and testable arrays

    def _shortens_train_test_arrays(self):
        half_test_length = len(self.x_test) // 2
        self.x_test = self.x_test[:half_test_length]
        self.y_test = self.y_test[:half_test_length]
        
        half_train_length = len(self.y_train) // 2
        self.x_train = self.x_train[:half_train_length]
        self.y_train = self.y_train[:half_train_length]

        
    def print_layer_trainable():
        for layer in conv_model.layers:
            print("{0}:\t{1}".format(layer.trainable, layer.name))

    
    def fine_tuning(self):
        transfer_layer = None
        for layer in self.model.layers:
            if layer.name == "flatten":
                break
            transfer_layer = layer

        # Extract that part of the brain which is convolutional
        conv_model = KModel(inputs = self.model.input, outputs = transfer_layer.output)

        # New model to replace the previous one xd
        self.model = models.Sequential()

        # Add the convolutional part of the brain
        self.model.add(conv_model)
        self.model.add(layers.Flatten())  # squizzy to one dimension
        self.model.add(layers.Dense(self.class_output_neurons * 20, activation = 'relu'))

        # Add a dropout-layer which may prevent overfitting and
        # improve generalization ability to unseen data
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(self.class_output_neurons, activation = "softmax"))  # Output layer

        # Put all the conv layers in false
        conv_model.trainable = False
        for layer in conv_model.layers:
            layer.trainable = False

        # Finally retrain the model
        self.train()

        
    
