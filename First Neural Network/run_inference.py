import requests
requests.packages.urllib3.disable_warnings()
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


#Fashion mnist Dataset - Pixel data of clothing articles

#import tensorflow and keras library
import tensorflow as tf
from tensorflow import keras

#import helper libraries
import numpy as np
import matplotlib.pyplot as plt

#load the dataset
fashion_mnist = keras.datasets.fashion_mnist

#separates the dataset into training examples and testing examples
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = tf.keras.models.load_model("my_first_model")
print(model.summary())

def runInference(choice):
	image = test_images[choice]
	label = test_labels[choice]

	prediction = model.predict(np.array([image]))
	index_of_max = np.argmax(prediction)
	predicted_class = class_names[index_of_max]
	expected_class = class_names[label]

	showImage(image, predicted_class, expected_class)

def showImage(image, predicted_class, expected_class):
	plt.figure()
	plt.imshow(image)
	plt.title("Predicted Class: " + predicted_class + " | " + "Expected Class: " + expected_class)
	plt.colorbar()
	plt.grid(False)
	plt.show()

while True:
	print("")
	print("The dataset contains 10000 images.")
	choice = input("Choose a number between 0 and 9999: ")

	if choice.isdigit():
		choice = int(choice)
		if choice >= 0 and choice <= 9999:
			runInference(choice)
		else:
			print("Choice not in range.")
			print("")







