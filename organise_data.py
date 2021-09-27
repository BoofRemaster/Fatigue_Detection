import os
import shutil
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img

def label_img(img, original_path, target_path_open, target_path_closed):
    word_label = img.split('_')[4]
    if word_label[0] == '1': # eyes open
        original_path = original_path + img
        target_path_open = target_path_open + img
        shutil.copyfile(original_path, target_path_open)
    elif word_label[0] == '0': # eyes closed
        original_path = original_path + img
        target_path_closed = target_path_closed + img
        shutil.copyfile(original_path, target_path_closed)

def main():
    # training data
    #
    original_path_train = '../FatigueDetection/eye_dataset/Train_Data/'
    train_target_path_open = '../FatigueDetection/eye_dataset/train_set/open/'
    train_target_path_closed = '../FatigueDetection/eye_dataset/train_set/closed/'

    for img in os.listdir(original_path_train):
        label_img(img, original_path_train , train_target_path_open, train_target_path_closed)

    # testing data
    #
    original_path_test = '../FatigueDetection/eye_dataset/Test_Data/'
    test_target_path_open = '../FatigueDetection/eye_dataset/test_set/open/'
    test_target_path_closed = '../FatigueDetection/eye_dataset/test_set/closed/'

    for img in os.listdir(original_path_test):
        label_img(img, original_path_test , test_target_path_open, test_target_path_closed)

# following code is used to view the data to ensure it is being read properly
#
def show_data(folder_path, data, oc, picture_size):
    plt.figure(figsize=(12,12))
    for i in range(1, 10, 1):
        plt.subplot(3, 3, i)
        img = load_img(folder_path+data+'/'+oc+'/'+os.listdir(folder_path+data+'/'+oc)[i], target_size=(picture_size, picture_size))
        plt.imshow(img)
    plt.show()

# show_data(folder_path, test_data, open, picture_size) # change parameters to see different data

# run program
main()