import pandas as pd
import re
from transformers import pipeline


path = 'C:/Users/Umer/Downloads/testdata.csv'

df = pd.read_csv(path, header=None, names=['col1', 'col2', 'col3', 'col4', 'User_id', 'Tweet'])

data = df[[ 'User_id', 'Tweet']]

#data.to_csv('output.csv', index=False, header=False)

#function to remove urls, usernames, utranslatable characters and converting to lower case
def preprocess_tweet(text):
    
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

#Passing each value to the preprocess_tweet and storing in new column
data['preprocessed_tweets'] = df['Tweet'].apply(preprocess_tweet)

#choosing the required trained model 
sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

sentiment = []

#Storing the sentiments as POS/NEG/NEU in sentiment list
for text in data['preprocessed_tweets'] :
    result = sentiment_pipeline(text)[0]
    sentiment.append(result['label'])

data['sentiment'] = sentiment
print(data)
data.to_csv('output.csv', index=False)
