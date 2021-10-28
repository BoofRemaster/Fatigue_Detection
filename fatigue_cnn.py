from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import metrics
import numpy as np

img_size = 48  # x/y dimensions to resize images to
batch_size = 64  # batch size for feeding CNN
num_epochs = 1
num_train_steps = 30  # train steps per epoch
num_val_steps = 30  # validation steps per epoch
train_data_dir = "./eye_dataset/train_set"
validation_data_dir = "./eye_dataset/test_set"


def generate_model():
    # setup training and validation image streams and sets
    train_gen = ImageDataGenerator(rescale=1. / 255)
    validation_gen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_gen.flow_from_directory(train_data_dir,
                                                    target_size=(img_size, img_size),
                                                    batch_size=batch_size,
                                                    color_mode='grayscale',
                                                    class_mode='binary')
    validation_generator = validation_gen.flow_from_directory(validation_data_dir,
                                                              target_size=(img_size, img_size),
                                                              batch_size=batch_size,
                                                              color_mode='grayscale',
                                                              class_mode='binary')

    # initialise sequential model
    model = Sequential()

    # add first spatial convolution layer
    model.add(Conv2D(filters=96, input_shape=(img_size, img_size, 1), kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu'))
    # add first downsampling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # add second and final spatial convolution layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    # add second and final downsampling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # add flatten layer
    model.add(Flatten())
    # add dense connection layer
    model.add(Dense(4096, input_shape=(256,), activation='relu'))
    # add dropout layer with rate of 0.4 to help prevent overfitting
    model.add(Dropout(0.4))

    # add second dense connection layer
    model.add(Dense(4096, activation='relu'))
    # add second dropout layer with rate of 0.4 to help prevent overfitting
    model.add(Dropout(0.4))

    # add final dense connection layer with single output node and sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # compile model with adam optimizer using binary crossentropy for loss function
    model.compile(optimizer=adam_v2.Adam(0.001), loss='binary_crossentropy', metrics=[metrics.BinaryAccuracy(), metrics.Recall(), metrics.AUC()])

    # fit the model to training data set and evaluate progressively with validation data set
    history = model.fit_generator(train_generator,
                        steps_per_epoch=num_train_steps,
                        epochs=num_epochs,
                        validation_data=validation_generator,
                        validation_steps=num_val_steps)

    np.save('model_history.npy', history.history)

    # return the fitted model
    return model


def load_model():
    history = np.load('model_history.npy', allow_pickle=True).item()
    print(history)


if __name__ == "__main__":
    generate_model().save("./output/model.h5")
    load_model()
