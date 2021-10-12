from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

img_size=224
batch_size=64
num_epochs=5
num_train_steps=30
num_val_steps=30
train_data_dir = "./eye_dataset/train_set"
validation_data_dir = "./eye_dataset/test_set"


def generate_model():
    train_gen = ImageDataGenerator(rescale=1. / 255)

    validation_gen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_gen.flow_from_directory(train_data_dir,
                                                    target_size = (img_size, img_size),
                                                    batch_size=batch_size,
                                                    color_mode='grayscale',
                                                    class_mode='binary')

    validation_generator = validation_gen.flow_from_directory(validation_data_dir,
                                                              target_size = (img_size, img_size),
                                                              batch_size=batch_size,
                                                              color_mode='grayscale',
                                                              class_mode='binary')

    model = Sequential()

    model.add(Conv2D(filters=96, input_shape=(img_size, img_size, 1), kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(4096, input_shape=(256,), activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=adam_v2.Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_generator,
                        steps_per_epoch=num_train_steps,
                        epochs=num_epochs,
                        validation_data=validation_generator,
                        validation_steps=num_val_steps)

    return model


if __name__ == "__main__":
    generate_model().save("./output/model.h5")
