import pandas as pd
from collections import deque
import random
import numpy as np
from sklearn import preprocessing

import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint


SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "LTC-USD"


def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def preprocess_df(df, SEQ_LEN):
    df = df.drop("future", 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.


    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!


'''
main_df = pd.DataFrame() # begin empty


ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider
for ratio in ratios:  # begin iteration
    print(ratio)
    dataset = f'crypto_data/{ratio}.csv'  # get the full path to the file.
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
#print(main_df.head())  # how did we do??

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

#print(main_df.head())

times = sorted(main_df.index.values)  # get the times
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]  # get the last 5% of the times

validation_main_df = main_df[(main_df.index >= last_5pct)]  # make the validation data where the index is in the last 5%
main_df = main_df[(main_df.index < last_5pct)]  # now the main_df is all the data up to the last 5%
'''

main_df = pd.DataFrame() # begin empty

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider
for ratio in ratios:  # begin iteration

    ratio = ratio.split('.csv')[0]  # split away the ticker from the file-name
    dataset = f'crypto_data/{ratio}.csv'  # get the full path to the file.
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)


main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
#print(main_df.head())  # how did we do??


RATIOS_TO_PREDICT = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]
EPOCHS = [6, 10, 20]
BATCH_SIZES = [32, 64, 128]
SEQ_LENS = [30, 60, 120]
LEARNING_RATES = [0.0001, 0.001, 0.01]


for coin in RATIOS_TO_PREDICT:
    for seq in SEQ_LENS:
        for batch_size in BATCH_SIZES:
            for epoch_size in EPOCHS:
                
                NAME = f"{RATIO}-RATIO-{seq}-SEQ-{batch_size}-BATCH-{epoch_size}-EPOCHS-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
                #NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

                main_df['future'] = main_df[f'{coin}_close'].shift(-FUTURE_PERIOD_PREDICT)
                main_df['target'] = list(map(classify, main_df[f'{coin}_close'], main_df['future']))

                main_df.dropna(inplace=True)

                ## here, split away some slice of the future data from the main main_df.
                times = sorted(main_df.index.values)
                last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

                validation_main_df = main_df[(main_df.index >= last_5pct)]
                main_df = main_df[(main_df.index < last_5pct)]


                train_x, train_y = preprocess_df(main_df, seq)
                validation_x, validation_y = preprocess_df(validation_main_df, seq)

                print(f"train data: {len(train_x)} validation: {len(validation_x)}")
                print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
                print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")


                model = Sequential()
                model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                model.add(CuDNNLSTM(128, return_sequences=True))
                model.add(Dropout(0.1))
                model.add(BatchNormalization())

                model.add(CuDNNLSTM(128))
                model.add(Dropout(0.2))
                model.add(BatchNormalization())

                model.add(Dense(32, activation='relu'))
                model.add(Dropout(0.2))


                # try this sometime later
                '''
                model.add(CuDNNLSTM(32))
                model.add(Dropout(0.1))
                model.add(BatchNormalization())

                model.add(Dense(8, activation='relu'))
                model.add(Dropout(0.2))
                '''

                model.add(Dense(2, activation='softmax'))


                for learn_rate in LEARNING_RATES:
                    opt = tf.keras.optimizers.Adam(lr=learn_rate, decay=1e-6)

                    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

                    tb = TensorBoard(log_dir="logs/{}".format(NAME))

                    # where to store all the epochs
                    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"

                    # the best model
                    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))


                    history = model.fit(
                        train_x, train_y,
                        batch_size=batch_size,
                        epochs=epoch_size,
                        validation_data= (validation_x, validation_y),
                        callbacks=[tb, checkpoint]
                    )

                    # Score model
                    score = model.evaluate(validation_x, validation_y, verbose=0)
                    print('Test loss:', score[0])
                    print('Test accuracy:', score[1])
                    # Save model
                    model.save("models/{}".format(NAME))