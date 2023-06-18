import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


def buildLSTM(config):
    visible = Input(shape=(config.input_layer_size, 1))
    hidden1 = LSTM(config.layer_1_lstm_size, return_sequences=True)(visible)
    hidden2 = Dropout(0.2)(hidden1)
    hidden3 = LSTM(config.layer_2_lstm_size, return_sequences=False)(hidden2)
    hidden4 = Dropout(0.2)(hidden3)
    hidden5 = Dense(config.layer_3_dense_size, activation=config.layer_3_activation)(hidden4)
    output = Dense(config.output_layer_size, activation=config.output_activation)(hidden5)
    model = tf.keras.Model(inputs=visible, outputs=output)

    # summarize layers
    print(model.summary())
    model.compile(optimizer=config.optimizer,
                  loss=config.loss,
                  metrics=config.metric)
    return model


def getRandomNumberBetween(min=-1, max=1):
    return random.random() * (max - min) + min


def buildLinearDataset(inputSize, outputSize):
    m = getRandomNumberBetween(-1, 1)
    c = getRandomNumberBetween(-1, 1)
    return np.array([m*i + c for i in range(inputSize + outputSize)])


def buildQuadraticDataset(inputSize, outputSize):
    order = random.randint(1, 5)
    c = [getRandomNumberBetween(-1,1) for _ in range(order)]
    intercept = getRandomNumberBetween(-1, 1)
    poly = lambda x: sum([c[i]*(x**i) for i in range(order)]) + intercept
    return np.array([poly(i) for i in range(inputSize + outputSize)])


def buildTrigonometricDataset(inputSize, outputSize):
    mX, mY = getRandomNumberBetween(-1, 1), getRandomNumberBetween(-1, 1)
    cX, cY = getRandomNumberBetween(-1, 1), getRandomNumberBetween(-1, 1)
    return np.array([mY*np.sin(mX*i + cX) + cY for i in range(inputSize + outputSize)])


def pickRandomDataset(inputSize, outputSize):
    choice = random.randint(0, 2)
    if choice == 0:
        return buildLinearDataset(inputSize, outputSize)
    elif choice == 1:
        return buildQuadraticDataset(inputSize, outputSize)
    else:
        return buildTrigonometricDataset(inputSize, outputSize)


def superimposeFunctions(inputSize, outputSize, minFunctions, maxFunctions):
    numFunctions = random.randint(minFunctions, maxFunctions)
    functions = [pickRandomDataset(inputSize, outputSize) for _ in range(numFunctions)]
    superFunc = functions[0]
    for i in range(1, numFunctions):
        if random.random() > 0.5:
            superFunc = superFunc + functions[i]
        else:
            superFunc = superFunc * functions[i]
    return superFunc


def buildFunctionDataset(config, superimpose=None):
    # Build the dataset, standardise it and split into train and test
    inputSize, outputSize = config.input_layer_size, config.output_layer_size
    size = config.dataset_size
    split = config.train_test_split
    if superimpose is not None:
        rawSet = np.array([superimposeFunctions(inputSize, outputSize, minFunctions=superimpose[0], maxFunctions=superimpose[1])
                           for _ in range(size)])
    else:
        rawSet = np.array([pickRandomDataset(inputSize, outputSize) for _ in range(size)])
    rawSet = rawSet - np.min(rawSet, axis=1, keepdims=True)
    rawSet = rawSet / (np.max(rawSet, axis=1, keepdims=True) + 10e-10)
    trainX, trainY = rawSet[:int(size * split), :inputSize], rawSet[:int(size * split), inputSize:]
    testX, testY   = rawSet[int(size * split):, :inputSize], rawSet[int(size * split):, inputSize:]
    return trainX, trainY, testX, testY


def plotInputOutput(xInput, yPred):
    plt.plot(np.arange(0, xInput.shape[0]), xInput, color='blue')
    plt.plot(np.arange(xInput.shape[0], xInput.shape[0]+yPred.shape[0]), yPred, color='red')
    plt.show()


def main():
    # Define config
    wandb.init(
        project="lstm-template",
        config={
            "dataset_size"      : 10000,
            "train_test_split"  : 0.8,
            "input_layer_size"  : 128,
            "layer_1_lstm_size" : 32,
            "layer_2_lstm_size" : 32,
            "layer_3_dense_size": 32,
            "layer_3_activation": "relu",
            "output_layer_size" : 32,
            "output_activation" : "sigmoid",
            "optimizer"         : "adam",
            "loss"              : "mse",
            "metric"            : "accuracy",
            "epochs"            : 5,
            "batch_size"        : 256
        }
    )
    config = wandb.config

    # Create test datasest
    trainX, trainY, testX, testY = buildFunctionDataset(config, superimpose=(3,8))

    # Build and train model
    model = buildLSTM(config)
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

    for i in range(20):
        pred = model.predict(testX[i].reshape(1, config.input_layer_size, 1))
        plotInputOutput(testX[i], pred.reshape(config.output_layer_size, 1))


if __name__ == '__main__':
    main()