from pickletools import optimize
import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    working_dir = os.getcwd()
    img_dirs = os.listdir(data_dir)

    for dir in img_dirs:

        dir_path = os.path.join(working_dir, data_dir, dir)
        imgages = os.listdir(dir_path)

        for img_file in imgages:
            
            img_file_path = os.path.join(dir_path, img_file)

            # read img as ndarray with shape (30,30,3)
            img = cv2.imread(img_file_path, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # append image to loaded dataset
            images.append(img)
            labels.append(dir)

    return (images, labels)
        


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    input_shape = ((IMG_WIDTH, IMG_HEIGHT, 3))
    output_shape = NUM_CATEGORIES

    # Define the model

    model = tf.keras.Sequential([

        # Convolution and Pooling 1
        tf.keras.layers.Conv2D(filters=4, kernel_size=3, input_shape=input_shape),
        tf.keras.layers.LeakyReLU(alpha=0.15),
        tf.keras.layers.BatchNormalization(momentum=0.8),

        tf.keras.layers.Conv2D(filters=4, kernel_size=3),
        tf.keras.layers.LeakyReLU(alpha=0.15),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        # Convolution and Pooling 2
        tf.keras.layers.Conv2D(filters=8, kernel_size=3),
        tf.keras.layers.LeakyReLU(alpha=0.15),
        tf.keras.layers.BatchNormalization(momentum=0.8),

        tf.keras.layers.Conv2D(filters=8, kernel_size=2),
        tf.keras.layers.LeakyReLU(alpha=0.15),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

        tf.keras.layers.Flatten(),

        # Deep Neural Net
        tf.keras.layers.Dense(64),
        tf.keras.layers.LeakyReLU(alpha=0.15),

        tf.keras.layers.Dense(output_shape),
        tf.keras.layers.LeakyReLU(alpha=0.15),

        # Probability Distribution over classes as output
        tf.keras.layers.Dense(output_shape, activation="softmax")
    ])

    # Compile the model
    model.compile(

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = ["accuracy"]
    )
    
    # print(model.summary())

    return model


if __name__ == "__main__":
    main()
