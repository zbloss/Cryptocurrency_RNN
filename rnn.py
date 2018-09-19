import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque
import random

ltc = pd.read_csv("crypto_data/LTC-USD.csv", names=['time', 'low', 'high', 'open', 'close', 'volume'])

print(ltc.head())

df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"] 

for ratio in ratios:
    print(ratio)

    dataset = f'crypto_data/{ratio}.csv'
    tmp_df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])

    tmp_df.rename(columns={"close": f'{ratio}_close', "volume": f'{ratio}_volume'}, inplace=True)

    tmp_df.set_index("time", inplace=True)
    tmp_df = tmp_df[[f'{ratio}_close', f'{ratio}_volume']]

    if len(df) == 0:
        df = tmp_df
    else:
        df = df.join(tmp_df)

df.fillna(method="ffill", inplace=True) # using previously known values for missing values
df.dropna(inplace=True)

df.to_csv('./crypto_data/cleaned_data.csv', index=None)

print(df.head())

df = pd.read_csv('./crypto_data/cleaned_data.csv')
print('cleaned_data read')
print(df.head(10))

# preceeding sequence length to grab for the RNN
SEQ_LEN = 60
# how far into the future we are trying to predict
FUTURE_PERIOD_PREDICT = 3
# Ratio we are trying to predict
RATIO_TO_PREDICT = "LTC-USD"

# basically, we are making a classification problem. If we predict the price to go up in 3 minutes then we buy, if we predict it to go down in 3 minutes than we don't buy or maybe we sell.

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

df['future'] = df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
# neat trick, the shift function goes down in the dataset, but making the param negative will go up that many rows. So, here we are grabbing the "LTC-USD" 3 minutes into the future.

df['target'] = list(map(classify, df[f'{RATIO_TO_PREDICT}_close'], df['future']))

print('\n\n\n')
print(df.head())

# grabbing the times column sorted in order
times = sorted(df.index.values) 

#grabbing the last 5% of the times
last_5pct = sorted(df.index.values)[-int(0.05*len(times))]

# this is our testing/validation data containing only the most recent 5% of the data
validation_df = df[(df.index >= last_5pct)]

# here is the training data, containing the other 95% of the data
df = df[(df.index < last_5pct)]

def preprocess_df(df):
    df = df.drop("future", axis=1)

    # lets normalize all columns except for the target
    for col in df.columns:
        if col != "target":

            # this makes each column a percentage rather than a value.
            # This normalizes the values comparing BTC etc. so we see the percentage change rather than the $$$ change
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)

            # now we scale everthing from 0-1
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    # here is the list that will contain the sequences
    sequential_data = []

    # these are our sequences. deque will allow newer values to enter while kicking out older values
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df:
        # storing everything but the target col
        prev_days.append([n for n in i[:-1]])
        
        # making sure we have the correct Sequence length
        if len(prev_days) == SEQ_LEN:

            # adding the most recent sequence to our list of sequences
            sequential_data.append([np.array(prev_days), i[-1]])
    
    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        
        # if it is a 'not buy'
        if target == 0:
            sells.append([seq, target])
        # check if it's a 'buy'
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    # which has less?
    lower = min(len(buys), len(sells))
    
    # making sure each list is only as long as the shortest list
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells

    # we can't feed all of the 1's then all of the 0's to the model, so we shuffle away!
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

X_train, y_train = preprocess_df(df)
X_test, y_test = preprocess_df(validation_df)

print('train data: {}, validation: {}'.format(len(X_train), X_test))
print('Don\'t buys: {}, Buys: {}'.format(y_train.count(0), y_train.count(1)))
print('VALIDATION Don\'t buys: {}, buys: {}'.format(y_test.count(0), y_test.count(1)))