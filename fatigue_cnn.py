from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

class FatigueCNN:
    def build(self):
        no_classes = 2
        model = Sequential()
        # 1st CNN layer
        model = self.add_layer(model, 64, (3,3), 'same', (48, 48, 1), 'relu', (2,2), 0.25)

        # 2nd CNN layer
        model = self.add_layer(model, 128, (5,5), 'same', None, 'relu', (2,2), 0.25)
        
        # 3rd CNN layer
        model = self.add_layer(model, 512, (3,3), 'same', None, 'relu', (2,2), 0.25)

        # 4th CNN layer
        model = self.add_layer(model, 512, (3,3), 'same', None, 'relu', (2,2), 0.25)

        model.add(Flatten())

        # 1st layer fully connected
        model = self.fully_connect(model, 256, 'relu', 0.25)

        # 2nd layer fully connected
        model = self.fully_connect(model, 512, 'relu', 0.25)
        
        model.add(Dense(no_classes, activation='softmax'))
        
        opt = adam_v2.Adam(learning_rate=0.0001) #apply optimisations to the model
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
    
    def trainCNN(self, model, out_dir, epochs, train_set, test_set):
        checkpoint = ModelCheckpoint(out_dir, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001)
        callbacks_list = [early_stopping, checkpoint, reduce_learning_rate]
        epochs = epochs
        trained_model = model.fit_generator(generator=train_set,
                            steps_per_epoch=train_set.n//train_set.batch_size,
                            epochs=epochs,
                            validation_data=test_set,
                            validation_steps=test_set.n//test_set.batch_size,
                            callbacks=callbacks_list
                        )
        return trained_model
    
    def plot_model(self, trained_model):
        plt.style.use('dark_background')

        plt.figure(figsize=(20,10))
        plt.subplot(1, 2, 1)
        plt.suptitle('Optimizer: Adam', fontsize=10)
        plt.ylabel('Loss', fontsize=16)
        plt.plot(trained_model.trained_model['loss'], label='Validation Loss')
        plt.plot(trained_model.trained_model['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        plt.ylabel('Accuracy', fontsize=16)
        plt.plot(trained_model.trained_model['accuracy'], label='Training Accuracy')
        plt.plot(trained_model.trained_model['val_accuracy'], label='Accuracry')
        plt.legend(loc='upper right')
        plt.show()
    
    def add_layer(self, model, filter, kernel_size, padding, input_shape, activation, pool_size, dropout):
        if input_shape is None:
            model.add(Conv2D(filter, kernel_size, padding=padding))
        else:
            model.add(Conv2D(filter, kernel_size, padding=padding, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(dropout))
        return model
    
    def fully_connect(self, model, dense_units, activation, dropout):
        model.add(Dense(dense_units))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(dropout))
        return model

class GetSets:
    def __init__(self, picture_size, folder_path, test_path, train_path, batch_size):
        self.picture_size = picture_size
        self.folder_path = folder_path
        self.test_path = test_path
        self.train_path = train_path
        self.batch_size = batch_size
        
    def get_set(self, gen_set, path, color_mode, class_mode, shuffle):
        set = gen_set.flow_from_directory(
            self.folder_path+path,
            target_size=(self.picture_size, self.picture_size),
            color_mode=color_mode,
            batch_size=self.batch_size,
            class_mode=class_mode,
            shuffle=shuffle
        )
        return set

def main():
    gen_train = ImageDataGenerator()
    gen_test = ImageDataGenerator()
    gs = GetSets(48, '../FatigueDetection/eye_dataset/', 'test_set', 'train_set', 128)
    train_set = gs.get_set(gen_train, gs.train_path, 'grayscale', 'categorical', True)
    test_set = gs.get_set(gen_train, gs.test_path, 'grayscale', 'categorical', False)

    fcnn = FatigueCNN()
    model = fcnn.build()
    trained_model = fcnn.trainCNN(model, "F:/Personal/Coding/Projects/FatigueDetection/output/model.h5", 5, train_set, test_set)
    fcnn.plot_model(trained_model)

# run program
main()