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
#We have 60000 images each one is 28 pixels x 28 pixels = 784 pixels
print("Dataset structure:", train_images.shape)
#We can see in what data type the data set comes in.
print("Data structure saving the dataset: ", type(train_images)) 
#Let's see the value of one pixel in the first image of the dataset
#This is the grayscale value of one pixel.
print("The value of one pixel:", train_images[0, 23, 23])
#The labels in the dataset are integers ranging from 0-9 and each integer represents a specific article of clothing.
print("Training labels: ", train_labels[:10])
#Create array of label names to indicate which is which
class_names = ["T-shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#Let's look at some images in the dataset
#plt.figure()
#plt.imshow(train_images[1])
#plt.colorbar()
#plt.grid(False)
#plt.show() #Uncomment here to show

#Preprocessing of training data and testing data
#We get values between 0 and 1
#Smaler values will make it easier for the model to process our values
train_images = train_images / 255.0
test_images = test_images / 255.0

#Creating the model
#This is the architecture of the Neural Network.
model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)), #Input layer of the NN
	keras.layers.Dense(128, activation="relu"), #Hidden layer of the NN
	keras.layers.Dense(10, activation="softmax") #Output layer of the NN
])

#INPUT LAYER
#The input layer will consist of 784 neurons. 
#Flatten means that our layer will reshape the 28x28 matrix into a vector of 784 neurons so that each pixel will be associated with one neuron.

#HIDDEN LAYER
#Dense denotes that this layer will be fully connected and that each neuron from the previous layer connects to each neuron of the next layer.
#The hidden layer has 128 neurons and it uses the rectify linear unit activation function - RELU.

#OUTPUT LAYER
#It has 10 neurons that determine the output of the model.
#Each neuron represents the probability of a given image being one of the 10 different classes.
#The activation function softmax is used on this layer to calculate the probability distribution of each class.
#The value of any neuron in this layer will be between 0 and 1, where 1 represents a high probability of the image being in that class.

#From WIKIPEDIA - "The softmax function is often used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes."

#HYPERPARAMETERS TUNNING
#Unlike the weights and the bias that we can not control, the hyperparameters we can manually change and try different combinations to see what works best for our dataset and NN architecture.

model.compile(
	optimizer="adam",
	loss="sparse_categorical_crossentropy",
	metrics=["accuracy"]
)

#ADAM is our OPTIMIZATION ALGORITHM.
#This can mean the difference between obtaining good results in minutes, hours and days.

#The Adam optimization algorithm is an extension to Stochastic Gradient Descent that has seen broader adoption for Deep Learning application in Computer Vision and Natural Language Processing.
#ADVANTAGES:
# - Computationally efficient
# - Little Memory requirements.
# - Well suited for problrmas that are large in terms of data and parameters.
# - Appropriate for problems with very noise or sparse gradients.
# - Hyperparameters have intuitive interpretaton and typically require little tuning.

#The most beneficial nature of the Adam optimization is the dapatative learning rate. It can compute adaptative learning rates for different parameters.

#SPARSE CATEGORICAL CROSSENTROPY computes the crossentropy between the labels and the predictions obtained.
#Measures the performance of the clasification model whose output is a probability between 0 and 1.
#Cross entropy increases as the predicted probability of the exaple diverges from the actual value.
#https://subscription.packtpub.com/book/big_data_and_business_intelligence/9781789132212/3/ch03lvl1sec30/understanding-categorical-cross-entropy-loss

#metrics=["accuracy"] how often prediction equal the correct labels.

#TRAINING THE MODEL
model.fit(train_images, train_labels, epochs=10)
#An epoch is a full iteration over the training examples.
#It can affect directly the results of the training process.

#EVALUATING THE MODEL
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print("Test accuracy: ", test_accuracy)

#MAKING PREDICTIONS
#model.predict receives an array of images.
predictions = model.predict(test_images)
#predictions is an array of arrays where each array represents a probability distribution.
print(predictions[0]) #This is result probability distribution for the first image.

#What is the result class?
#Using this numpy function we can get the index of the maximum value in the array.
index_of_max = np.argmax(predictions[0]) #Change here!
print(index_of_max)

#Print output class and show image.
output_class = class_names[index_of_max]
print(output_class)

plt.figure()
plt.imshow(test_images[0]) #Change here!
plt.colorbar()
plt.grid(False)
plt.show()

#SAVE THE MODEL
model.save("my_first_model")


























