from keras.datasets import mnist 		# subroutines for fetching the MNIST dataset
from keras.models import Model 			# basic class for specifying and training a neural network
from keras.layers import Input, Dense 	# the 2 types of neural network layer will be used
from keras.utils import np_utils 		# utilites for one-hot encodingof GT values

batch_size = 128
num_epochs = 20
hidden_size = 512

num_train = 60000
num_test = 10000

height, width, depth = 28, 28, 1
num_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(num_train, height * width) 		# flatten data to 1D
X_test = X_test.reshape(num_test, height * width)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255 # normalize data to [0,1] range
X_test /= 255

Y_train = np_utils.to_categorical(y_train, num_classes) 	# one-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes)

inp = Input(shape = (height * width,))
hidden_1 = Dense(hidden_size, activation = 'relu')(inp)
hidden_2 = Dense(hidden_size, activation = 'relu')(hidden_1)
out = Dense(num_classes, activation = 'softmax')(hidden_2)

model = Model(input = inp, output = out)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = num_epochs, verbose = 1, validation_split = 0.1)
model.save('my_model.h5')
print(model.evaluate(X_test, Y_test, verbose = 1))