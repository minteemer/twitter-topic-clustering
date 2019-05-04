# %%
import os
import pickle
import re
from pathlib import Path
from pprint import pprint

import en_core_web_md
import en_core_web_lg
import langdetect
import nltk
import pandas
import pandas as pd
import preprocessor as tweet_prep  # pip install tweet-preprocessor
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import spacy

DATA_PATH = Path("./data/")
PREPROCESSED_DATA_PATH = DATA_PATH / "cleaned"
RAW_DATA_PATH = DATA_PATH / "raw"
CACHE_FOLDER = Path("./cache/")

STOP_WORDS = stopwords.words("english")
DETECT_NER_LABELS = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]

tweet_prep.set_options(tweet_prep.OPT.URL, tweet_prep.OPT.MENTION, tweet_prep.OPT.HASHTAG, tweet_prep.OPT.RESERVED,
                       tweet_prep.OPT.EMOJI, tweet_prep.OPT.SMILEY, tweet_prep.OPT.NUMBER)

print("Loading model...")
nlp = en_core_web_lg.load()


def tokenize(text, vocabulary=None):
    return [token.lower()
            for token in nltk.word_tokenize(text)
            if token not in STOP_WORDS
            and (vocabulary is None or token in vocabulary)]


def extract_tags(text):
    parsed_tweet = tweet_prep.parse(text)
    hashtags = {h.match for h in parsed_tweet.hashtags} if parsed_tweet.hashtags else {}
    mentions = {m.match for m in parsed_tweet.mentions} if parsed_tweet.mentions else {}
    urls = {u.match for u in parsed_tweet.urls} if parsed_tweet.urls else {}
    return hashtags, mentions, urls


def preprocess_tweet(text):
    return tweet_prep.clean(BeautifulSoup(text, 'lxml').text)


def extract_entities(text):
    return {ent.text for ent in nlp(text).ents if ent.label_ in DETECT_NER_LABELS}


def get_topic_contents(topic_file_name):
    with open(RAW_DATA_PATH / topic_file_name, "r", encoding="utf-8") as topic_file:
        data = pandas.read_csv(topic_file, sep="\t", header=None,
                               index_col="tweet_id", names=["tweet_id", "user_name", "user_id", "tweet"])

    data = data.drop(columns=["user_name", "user_id"])  # Remove useless columns
    data = data[data["tweet"] != "Not Available"]  # Remove not available tweets
    # data = data[data["tweet"].apply(lambda t: langdetect.detect(t) == "en")]  # TODO: Leave only english tweets?

    data["hashtags"], data["mentions"], data["urls"] = zip(*data["tweet"].map(extract_tags))
    data["processed"] = data["tweet"].apply(preprocess_tweet)
    data["entities"] = data["processed"].apply(extract_entities)

    # data = data.drop_duplicates("processed")  #TODO: remove duplicated tweets?
    return data


def get_topics():
    with open(DATA_PATH / "TT-annotations.csv", "r", encoding="utf-8") as topic_file:
        data = pandas.read_csv(topic_file, sep=";", header=None, names=["hash", "date", "name", "type"])
    return data


def preprocess_all_topics():
    topics = get_topics()
    downloaded = set(os.listdir(RAW_DATA_PATH))
    topics = topics[topics["hash"].apply(lambda h: h in downloaded)]
    for index, topic in topics.iterrows():
        print(f"{index}/{topics.shape[0]}: Preprocessing {topic['name']} ({topic['hash']})")

        topic_data = get_topic_contents(topic['hash'])
        if topic_data is not None:
            topic_data["topic"] = topic['name']
            topic_data.to_pickle(PREPROCESSED_DATA_PATH / topic['hash'])


def get_preprocessed_topics():
    topics = get_topics()
    path = DATA_PATH / "cleaned"
    preprocessed = set(os.listdir(path))
    topics = topics[topics["hash"].apply(lambda h: h in preprocessed)]
    topics_data = [pd.read_pickle(path / topic['hash'])
                   for index, topic in topics.iterrows()]
    return pd.concat(topics_data)


if __name__ == '__main__':
    preprocess_all_topics()
