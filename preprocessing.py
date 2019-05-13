import csv
import os
from pathlib import Path
import en_core_web_lg
import pandas as pd
import preprocessor as tweet_prep
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')  # Ignore redundant bs4 warnings

DATA_PATH = Path("./data/")  # Data folder path
PREPROCESSED_DATA_PATH = DATA_PATH / "preprocessed"  # Preprocessed topics files path
RAW_DATA_PATH = DATA_PATH / "raw"  # Raw topics files path

STOP_WORDS = stopwords.words("english")
DETECT_NER_LABELS = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]

tweet_prep.set_options(tweet_prep.OPT.URL, tweet_prep.OPT.MENTION, tweet_prep.OPT.HASHTAG, tweet_prep.OPT.RESERVED,
                       tweet_prep.OPT.EMOJI, tweet_prep.OPT.SMILEY, tweet_prep.OPT.NUMBER)

model = None


def get_model():
    """ Lazy initializer of model """
    global model

    if model is None:
        print("Loading model...")
        model = en_core_web_lg.load()

    return model


def extract_references(text):
    parsed_tweet = tweet_prep.parse(text)
    hashtags = [h.match for h in parsed_tweet.hashtags] if parsed_tweet.hashtags else []
    mentions = [m.match for m in parsed_tweet.mentions] if parsed_tweet.mentions else []
    urls = [u.match for u in parsed_tweet.urls] if parsed_tweet.urls else []
    return hashtags, mentions, urls


def extract_entities(text):
    nlp = get_model()
    return [ent.text for ent in nlp(text).ents if ent.label_ in DETECT_NER_LABELS]


def preprocess_tweet(text):
    return tweet_prep.clean(BeautifulSoup(text, 'lxml').text)


def get_topic_contents(topic_file_name):
    data = pd.read_csv(RAW_DATA_PATH / topic_file_name, sep="\t", quoting=csv.QUOTE_NONE, header=None,
                       index_col="tweet_id", names=["tweet_id", "user_name", "user_id", "tweet"])

    data = data.drop(columns=["user_name", "user_id"])  # Remove useless columns
    data = data[data["tweet"] != "Not Available"]  # Remove not available tweets
    # data = data[data["tweet"].apply(lambda t: langdetect.detect(t) == "en")]  # TODO: Leave only english tweets?

    data["hashtags"], data["mentions"], data["urls"] = zip(*data["tweet"].map(extract_references))  # Extract references
    data["processed"] = data["tweet"].apply(preprocess_tweet)  # Clean tweets
    data["entities"] = data["processed"].apply(extract_entities)  # Extract named entities
    return data


def get_topics_list():
    return pd.read_csv(DATA_PATH / "TT-annotations.csv", sep=";",
                       header=None, names=["hash", "date", "name", "type"])


def join_references(tweet):
    return list(tweet.hashtags) + list(tweet.mentions) + list(tweet.urls) + list(tweet.entities)


def preprocess_all_topics():
    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)

    topics = get_topics_list()
    for index, topic in topics.iterrows():
        print(f"{index}/{topics.shape[0]}: Preprocessing {topic['name']} ({topic['hash']})")
        if os.path.isfile(PREPROCESSED_DATA_PATH / topic['hash']):
            continue

        topic_data = get_topic_contents(topic['hash'])
        if topic_data is not None:
            topic_data["topic"] = topic['name']
            topic_data["references"] = topic_data.apply(join_references, axis=1)
            topic_data.to_pickle(PREPROCESSED_DATA_PATH / topic['hash'])


def get_preprocessed_topics(n_topics=None, joined_references=True):
    topics = get_topics_list()

    if n_topics is not None:
        topics = topics[:n_topics]

    topics_data = [pd.read_pickle(PREPROCESSED_DATA_PATH / topic['hash'])
                   for index, topic in topics.iterrows()]
    data = pd.concat(topics_data, sort=False)
    if joined_references:
        data = data.drop(columns=["entities", "hashtags", "mentions", "urls"])
    else:
        data = data.drop(columns=["references"])
    return data


def topic_len_stat():
    lens = []
    topics = get_topics_list()
    for index, topic in topics.iterrows():
        topics_data = pd.read_pickle(PREPROCESSED_DATA_PATH / topic['hash'])
        lens.append(topics_data.shape[0])

    print(f"min: {min(lens)}, max:{max(lens)}")
    plt.hist(lens, bins=30)
    plt.title("Number of tweets in topic histogram")
    plt.show()


if __name__ == '__main__':
    preprocess_all_topics()
    # topic_len_stat()
