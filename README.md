# Table of Contents
* [Description](https://github.com/RaudelCasas1603/Image-Classification#frame-recognition-video)
* [How to install it?](https://github.com/RaudelCasas1603/Image-Classification#how-to-install-it)
* [How to train it?](https://github.com/RaudelCasas1603/Image-Classification#how-to-train-it)
   * [To collect your own data](https://github.com/RaudelCasas1603/Image-Classification#to-collect-your-own-data)
   * [To build your own dataset](https://github.com/RaudelCasas1603/Image-Classification#to-build-your-own-dataset)
   * [To train the CNN](https://github.com/RaudelCasas1603/Image-Classification#to-train-the-cnn)
   * [To fine tuning the CNN](https://github.com/RaudelCasas1603/Image-Classification#to-fine-tuning-the-cnn)
* [Test a simple Demo](https://github.com/RaudelCasas1603/Image-Classification#test-a-simple-demo)
* [Examples](https://github.com/RaudelCasas1603/Image-Classification#examples)
   * [Video example](https://github.com/RaudelCasas1603/Image-Classification#video-example) 
* [References](https://github.com/RaudelCasas1603/Image-Classification#references)

# Frame Recognition Video
The **Frame Recognition Video** project is a **Python-based application** that utilizes the [OpenCV](https://github.com/opencv/opencv-python) library for capturing and preprocessing video frames. It also incorporates a convolutional neural networks **(CNNs)** from [TensorFlow](https://github.com/tensorflow/tensorflow), a popular deep learning framework, to perform **advanced image classification tasks**.

# How to install it?
To install Frame Recognition Video application, follow these steps:
1. Clone the repository to your local machine:
   ```shell
   git clone git@github.com:RaudelCasas1603/Image-Classification.git
   ```
2. Navigate to the project directory:
   ```shell
   cd Image-Classification
   ```
3. Create a virtual environment (optional but recommended):
   ```shell
   python3 -m venv env
   ```
4. Activate the virtual environment:
   * On macOS and Linux:
     ```shell
     source env/bin/activate
     ```
   * On Windows:
      ```shell
      .\env\Scripts\activate
      ```
5. Install the package program:
   ```shell
   pip install .
   ```
That's it! You have now successfully installed the Frame Recognition Video application.
# How to train it?
## To collect your own data
You can create your collection of data, firstly make sure that you already have installed the program in your
python environment, then follow the next steps:
1. Open your terminal or command prompt.
2. Navigate to the directory where your eviroment is located. For example:
   ```shell
   cd  /path/to/your/env
   ```
3. Activate the virtual environment if you have created one (optional):
   ```shell
   source env/bin/activate  # On macOS and Linux
   .\env\Scripts\activate  # On Windows
   ```
4. Run the command with the **collect-data** command and specify the desired arguments:
   ```shell
    fmr collect-data -n 100 -d directory-to-store-data/ -c classes.json
   ```
   * The **-n 100** flag is optional and specifies that you want to generate 100 pictures per class or sign. By default, the program is programmed to take at least 1000 pictures.
   * The **-d directory-to-store-data/** flag is optional and sets the directory where the collected data will be stored. Replace directory-to-store-data/ with the actual directory path. If the directory doesn't exist, it will be created. By default, it uses the data/ directory.
   * The **-c classes.json** flag specifies a JSON file with each leabel to classify.
   
5. The command will start collecting the data based on the provided arguments. It will generate pictures for each class and store them in the specified folder.

6. Once the data collection is completed, you will see the message **"Data collection completed."** printed in the terminal.

That's it! You have successfully created a data collection using the collect-data command. Adjust the arguments as needed to customize your data collection process.
## To build your own dataset
Firstly make sure that you have collected your data, then to build your own dataset just follow the next steps:
1. Open your terminal or command prompt.
2. Navigate to the directory where your environment is located. For example:
   ```shell
   cd /path/to/your/env
   ```
3. Activate the virtual environment if you have created one (optional):
   ```shell
   source env/bin/activate  # On macOS and Linux
   .\env\Scripts\activate  # On Windows
   ```
4. Run the command with the **build-dataset** command and specify the desired arguments:
   ```shell
   fmr build-dataset -f dataset.pickle -d data/
   ```
   * **-f dataset.pickle** specifies the output filename of the built dataset. Replace dataset.pickle with the **desired filename**. By default, **data.pickle** is the output dataset filename.
   * **-d data/** sets the directory where the raw data is stored. Replace **data/** with the actual directory. By default, the raw data is stored in the data/ directory.
5. The command will start building the dataset based on the provided arguments. It will process the images from the specified data directory and generate the dataset file.
6. Once the dataset is built, you will see the message **"Dataset built."** printed in the terminal.

## To train the CNN
1. Open your terminal or command prompt.
2. Navigate to the directory where your environment is located. For example:
   ```shell
   cd /path/to/your/env
   ```
3. Activate the virtual environment if you have created one (optional):
   ```shell
   source env/bin/activate  # On macOS and Linux
   .\env\Scripts\activate  # On Windows
   ```
4. Run the command with the **train** flag:
   ```shell
   fmr train
   ```
5. To initiate model for retraining, you can load it by specifying the **-m** flag along with the **your_model.pickle** file:
  ```shell
  fmr train -m your_model.pickle
  ```
  By default, the program utilizes the file named **model.pickle**.
   
## To Fine-Tuning the CNN
1. Open your terminal or command prompt.
2. Navigate to the directory where your environment is located. For example:
   ```shell
   cd /path/to/your/env
   ```
3. Activate the virtual environment if you have created one (optional):
   ```shell
   source env/bin/activate  # On macOS and Linux
   .\env\Scripts\activate  # On Windows
   ```
4. Execute the command with the tune option, and if desired, use the optional flag **-m** to specify a specific model:
   ```shell
   fmr tune -m specific_model.pickle
   ```
   **Note**: Replace **specific_model** with the name of the **desired model** you want to select, by default, the program utilizes the file named **model.pickle**.
   
# Test a simple Demo
1. Open your terminal or command prompt.
2. Navigate to the directory where your environment is located. For example:
   ```shell
   cd /path/to/your/env
   ```
3. Activate the virtual environment if you have created one (optional):
   ```shell
   source env/bin/activate  # On macOS and Linux
   .\env\Scripts\activate  # On Windows
   ```
4. Run the command with the **transcript** command and specify the desired arguments:
   ```shell
   fmr transcript -c classes.json
   ```
    * The **-c classes.json** flag specifies a JSON file with each leabel to classify.
   
   Using this command, you will be able to obtain transcriptions for each label based on what the **CNN** sees.

# Examples
I trained to recognize me in real time at my desk :).
## The summary of the model
![image](https://github.com/RaudelCasas1603/Image-Classification/assets/66882463/0cf2f46c-91ee-4982-8326-aed85984ec24)

## The trained output
![image](https://github.com/RaudelCasas1603/Image-Classification/assets/66882463/c63508c6-e7cd-4e4c-aba7-f411396b55f4)

## Video example   

https://github.com/RaudelCasas1603/Image-Classification/assets/66882463/62b20e7d-a684-40e5-a516-396000509e1b

## This is a simple output of the transcription
![image](https://github.com/RaudelCasas1603/Image-Classification/assets/66882463/4cf8bd24-f7cf-4b65-86ff-397b4361c2cc)


# References
* Computer vision engineer. (2023, January 26). Sign language detection with Python and Scikit Learn | Landmark detection | Computer vision tutorial [Video]. YouTube. https://www.youtube.com/watch?v=MJCSjXepaAM
* Normalized Nerd. (2021, January 13). Decision Tree Classification Clearly Explained! [Video]. YouTube. https://www.youtube.com/watch?v=ZVR2Way4nwQ
* Normalized Nerd. (2021b, April 21). Random Forest Algorithm Clearly Explained! [Video]. YouTube. https://www.youtube.com/watch?v=v6VJ2RO66A
* Gutta, S. (2022, January 6). Folder Structure for Machine Learning Projects | by Surya Gutta  | Analytics Vidhya. Medium. https://medium.com/analytics-vidhya/folder-structure-for-machine-learning-projects-a7e451a8caaa
* Convert python opencv mat image to tensorflow image data. (n.d.). Stack Overflow. Retrieved May 19, 2023, from https://stackoverflow.com/questions/40273109/convert-python-opencv-mat-image-to-tensorflow-image-data
* Convolutional neural network (CNN). (n.d.). TensorFlow. Retrieved May 19, 2023, from https://www.tensorflow.org/tutorials/images/cnn



