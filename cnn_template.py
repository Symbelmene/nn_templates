import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Flatten

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


def buildCNN(config):
    input_layer   = Input(shape=(config.input_layer_size[0], config.input_layer_size[1], 3))
    conv_layer_1  = Conv2D(filters=8, kernel_size=(3, 3), activation='relu')(input_layer)
    conv_layer_2  = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(conv_layer_1)
    conv_layer_3  = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv_layer_2)
    flatten_layer = Flatten()(conv_layer_3)
    dense_layer_1 = Dense(units=16, activation='relu')(flatten_layer)
    dense_layer_2 = Dense(units=8, activation='relu')(dense_layer_1)
    output_layer  = Dense(units=config.output_layer_size, activation='sigmoid')(dense_layer_2)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    model.compile(optimizer=config.optimizer,
                  loss=config.loss,
                  metrics=[config.metric])

    # Print model summary
    model.summary()

    return model


def addRectangleToImage(img, config):
    # Choose random color
    color = tuple([random.randint(0, 255) for _ in range(3)])

    # Choose random start and end point for rectangle
    start_point = (random.randint(0, config.input_layer_size[0] - 1),
                   random.randint(0, config.input_layer_size[1] - 1))
    end_point = (random.randint(start_point[0], config.input_layer_size[0] - 1),
                 random.randint(start_point[1], config.input_layer_size[1] - 1))

    # Draw rectangle
    img = cv2.rectangle(img, start_point, end_point, color, -1)
    return img


def addCircleToImage(img, config):
    # Choose random color
    color = tuple([random.randint(0, 255) for _ in range(3)])

    # Choose random center point and radius for circle
    center = (random.randint(0, config.input_layer_size[0] - 1),
              random.randint(0, config.input_layer_size[1] - 1))
    radius = random.randint(0, min(config.input_layer_size) - 1)

    # Draw circle
    img = cv2.circle(img, center, radius, color, -1)
    return img


def buildImageDataset(config):
    # Create empty arrays for images and labels
    images = []
    labels = []

    # Create a dataset of images with random shapes
    for i in range(config.dataset_size):
        img = np.zeros((config.input_layer_size[0], config.input_layer_size[1], 3))
        numShapes = random.randint(1, config.output_layer_size)
        for _ in range(numShapes):
            if random.random() < 0.5:
                img = addRectangleToImage(img, config)
            else:
                img = addCircleToImage(img, config)

        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        labels.append(numShapes)
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split into train and test sets
    split = int(config.dataset_size * config.train_test_split)
    trainX, trainY = images[:split], labels[:split]
    testX, testY   = images[split:], labels[split:]

    # Return dataset
    return trainX, trainY, testX, testY


def main():
    # Define config
    wandb.init(
        project="cnn-template",
        config={
            "dataset_size"       : 10000,
            "train_test_split"   : 0.8,
            "input_layer_size"   : (128, 128),
            "output_layer_size"  : 4,
            "output_activation"  : "softmax",
            "optimizer"          : "adam",
            "loss"               : "mse",
            "metric"             : "accuracy",
            "epochs"             : 5,
            "batch_size"         : 256
        }
    )
    config = wandb.config

    # Create test datasest
    trainX, trainY, testX, testY = buildImageDataset(config)

    # Build and train model
    model = buildCNN(config)
    history = model.fit(trainX, trainY,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        verbose=1,
                        validation_data=(testX, testY),
                        callbacks=[WandbMetricsLogger(log_freq=5),
                                   WandbModelCheckpoint("models")]),

    # Save model and plot predictions
    model.save('lstm_model.h5')
    wandb.finish()


if __name__ == '__main__':
    main()