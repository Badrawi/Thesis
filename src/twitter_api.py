import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import pandas as pd
import csv
import re #regular expression
from textblob import TextBlob
import string
#import preprocessor as p
from text_preprocessing import TextPreprocessing
import os
from config import config
class TwitterApi:

    textPreprocessing = TextPreprocessing()
    def __init__(self):
        auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)
        auth.set_access_token(config.access_key, config.access_secret)
        self.api = tweepy.API(auth)
        self.emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        self.COLS = ['original_text', 'clean_text', 'sentiment', 'polarity', 'subjectivity', 'lang','hashtags']
        emoticons_happy = set([
            ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
            ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
            '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
            'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
            '<3'
        ])
        emoticons_sad = set([
            ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
            ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
            ':c', ':{', '>:\\', ';('
        ])
        self.emoticons = emoticons_happy.union(emoticons_sad)

    def getTweets(self,keyword,file):
        #file = "../data/twitter.csv"
        if os.path.exists(file):
            df = pd.read_csv(file, header=0)
        else:
            df = pd.DataFrame(columns=self.COLS)
            # page attribute in tweepy.cursor and iteration
        for page in tweepy.Cursor(self.api.search, q=keyword,
                                  count=200, include_rts=False).pages(50):
            #print("new page")
            for status in page:
                new_entry = []
                status = status._json

                ## check whether the tweet is in english or skip to the next tweet
                if status['lang'] != 'en':
                    continue

                clean_text = status['text']
                blob = TextBlob(clean_text)
                Sentiment = blob.sentiment

                # seperate polarity and subjectivity in to two variables
                polarity = Sentiment.polarity
                subjectivity = Sentiment.subjectivity

                new_entry += [status['text'], clean_text, Sentiment, polarity, subjectivity,
                              status['lang']]
                hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
                new_entry.append(hashtags)
                single_tweet_df = pd.DataFrame([new_entry], columns=self.COLS)
                df = df.append(single_tweet_df, ignore_index=True)
                csvFile = open(file, 'a', encoding='utf-8')
            df.to_csv(csvFile, mode='a', columns=self.COLS, index=False, encoding="utf-8")

if __name__ == "__main__":
    file = "../data/fit.csv"
    stress_keyword = "(fitness OR healthy) AND (inshape OR [in shape])"
    twitterApi = TwitterApi()
    twitterApi.getTweets(stress_keyword,file)
