from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.layers.merge import average
from keras.utils import np_utils		#one-hto encoding
from keras.regularizers import l2		#L2-regularization
from keras.layers.normalization import BatchNormalization 
from keras.preprocessing.image import ImageDataGenerator #data augmentation
from keras.callbacks import EarlyStopping 

batch_size = 128
num_epochs = 50
kernel_size = 3
pool_size = 2
conv_depth = 32
drop_prob_1 = 0.25			#dropout after pooling with probability 0.25
drop_prob_2 = 0.5			# droput in the FC layer with probability 0.5
hidden_size = 128			# there will be 128 neurons in both hiddel layers
l2_lambda = 0.0001			# use 0.0001 as a L2-regularization factor
ens_models = 3				# will train 3 separate models on the data

num_train = 60000
num_test = 10000

height, width, depth = 28, 28, 1 # MNIST dataset
num_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], depth, height, width)
X_test = X_test.reshape(X_test.shape[0], depth, height, width)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = np_utils.to_categorical(y_train, num_classes)		#One-hot encode the labels
Y_test = np_utils.to_categorical(y_test, num_classes)

# Explicity split the training and validation sets
X_val = X_train[54000:]
Y_val = Y_train[54000:]
X_train = X_train[:54000]
Y_train = Y_train[:54000]

inp = Input(shape = (depth, height, width))
inp_norm = BatchNormalization(axis = 1)(inp)

outs = []		#the list of ensemble outputs

for i in range(ens_models):
	# Conv[32] -> Conv[32] -> Pool (with dropout on the pooling layer), applying BN in between
	conv1 = Convolution2D(conv_depth, kernel_size, kernel_size, border_mode = 'same', init = 'he_uniform', W_regularizer = l2(l2_lambda), activation = 'relu')(inp_norm)
	conv1 = BatchNormalization(axis = 1)(conv1)
	conv2 = Convolution2D(conv_depth, kernel_size, kernel_size, border_mode = 'same', init = 'he_uniform', W_regularizer = l2(l2_lambda), activation = 'relu')(conv1)
	conv2 = BatchNormalization(axis = 1)(conv2)
	pool_1 = MaxPooling2D(pool_size = (pool_size, pool_size), dim_ordering="th")(conv2)
	drop_1 = Dropout(drop_prob_1)(pool_1)
	flat = Flatten()(drop_1)
	hidden = Dense(hidden_size, init = 'he_uniform', W_regularizer = l2(l2_lambda), activation = 'relu')(flat)				# Hidden ReLU layer
	hidden = BatchNormalization(axis = 1)(hidden)
	drop = Dropout(drop_prob_2)(hidden)
	outs.append(Dense(num_classes, init = 'glorot_uniform', W_regularizer = l2(l2_lambda), activation = 'softmax')(drop))	# Output softmax layer

out = average(outs)			# Avarage the predictioins to obtain the final output

model = Model(input = inp, output = out)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

datagen = ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1)			# randomly shift images horizontally and vertically
datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size), samples_per_epoch = X_train.shape[0], nb_epoch = num_epochs, validation_data = (X_val, Y_val), verbose = 1,
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 5)])

res = model.evaluate(X_test, Y_test, verbose = 1)

print("Res = ", res)