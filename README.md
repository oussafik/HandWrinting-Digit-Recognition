# Handwritten Digit Recognition with GUI

This repository contains a script that demonstrates how to use a pre-trained convolutional neural network model for handwritten digit recognition in a Tkinter GUI application. The script utilizes the MNIST dataset, keras, PIL and OpenCV libraries.

## Scipts

- main.py: This script trains a convolutional neural network on the MNIST dataset and saves the trained model to an h5 file.

- application_1.py: This script is a Tkinter GUI application that allows the user to draw a digit on a canvas and predict the digit using the pre-trained model saved in the first script.

- application_2.py : This script is also a Tkinter GUI application that allows the user to upload an image of a digit and predict the digit using the pre-trained model saved in the first script.

## Requirements

- Python 3
- Tkinter
- PIL
- keras
- numpy
- OpenCV

## Usage

1. Clone the repository
2. In the command line, navigate to the directory where the scripts are located
3. Run the script that you want to use by using the command 'python script_name.py'


## Note

- Ensure that you have run the first script 'main.py' before running the other two scripts.
- The first script can be run using the command 'python main.py'
- The script 'application_2.py' needs OpenCV library to be installed in your environment in order to run it
- Ensure that Ghostscript is installed on your system and the path is correctly set in the script (This is important to run application_2.py).
You can download it from this link "https://www.ghostscript.com/download/gsdnld.html"


This project is just a demonstration of how to use a pre-trained model for prediction in a GUI application and can be further extended and customized for other use cases.



