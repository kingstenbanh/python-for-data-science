import tweepy
import csv
import numpy as np
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Step 1 - Tweeter API keys
consumer_key = 'bOedteGhR60TwJxRFfoN2wtJv'
consumer_secret = 'uIkFC8KHlVyPyypAsKwwzBB01uyLYA89Pvnth5m0EwZN008N1e'
access_token = '1641890911-bPmcV7YEUJ7BO3z7jqFV7Oe4qBEYLku3qDzOdZy'
access_token_secret = 'eboKGNIZpfJANaqzvOluUnVoYvJt4O9MO4B0qbjs2EfAA'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Step 2 - Search for your company name on Twitter
public_tweets = api.search('Facebook')

# Step 3 - Define a threshold for each sentiment to classify each
# as positive or negative. If the majority of tweets you've collected are positive
# then use your neural network to predict a future price
threshold = 0
pos_sent_tweet = 0
neg_sent_tweet = 0
for tweet in public_tweets:
    analysis = TextBlob(tweet.text)

    if analysis.sentiment.polarity >= threshold:
        pos_sent_tweet = pos_sent_tweet + 1
    else:  
        neg_sent_tweet = neg_sent_tweet + 1

if pos_sent_tweet > neg_sent_tweet:
    print('Overall Positive')
else:
    print('Overall Negative')

# Step 4 - Data collection
dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)

        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))

    return

# Step 5 - Reference your CSV file here
get_data('fb.csv')
plt.plot(prices)

plt.show()

def create_datasets(dates, prices):
    train_size = int(0.80 * len(dates))
    TrainX, TrainY = [], []
    TestX, TestY = [], []
    cntr = 0

    for date in dates:
        if cntr < train_size:
            TrainX.append(date)
        else:
            TestX.append(date)
    
    for price in prices:
        if cntr < train_size:
            TrainY.append(price)
        else:
            TestY.append(price)

    return TrainX, TrainY, TestX, TestY

# Step 6 - Build your neural network model, train 
def predict_price(dates, prices, x):
    TrainX, TrainY, TestX, TestY = create_datasets(dates, prices)

    TrainX = np.reshape(TrainX, (len(TrainX), 1))
    TrainY = np.reshape(TrainY, (len(TrainY), 1))
    TestX = np.reshape(TestX, (len(TestX), 1))
    TestY = np.reshape(TestY, (len(TestY), 1))
    
    model = Sequential()
    model.add(Dense(32, input_dim=1, init='uniform', activation='relu'))
    model.add(Dense(32, input_dim=1, init='uniform', activation='relu'))
    model.add(Dense(16, init='uniform', activation='relu'))

    model.add(Dense(1, init='uniform', activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(TrainX, TrainY, nb_epoch=100, batch_size=3, verbose=1)

predict_price(dates, prices, 2)

