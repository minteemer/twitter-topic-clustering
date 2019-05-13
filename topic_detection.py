from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words("english")

EXPLAINED_VARIANCE_CUMSUM_THRESHOLD = 0.18


def get_combined_tf_idf_matrix(data, retur_vocab=False):
    references = set()
    for tweet in data.itertuples():
        references.update(tweet.references)
    reference_indexes = {reference: index for index, reference in enumerate(references)}

    references_matrix = lil_matrix((data.shape[0], len(references)), dtype=int)
    for index, tweet in enumerate(data.itertuples()):
        for reference in tweet.references:
            references_matrix[index, reference_indexes[reference]] = 1

    count_vectorizer = CountVectorizer(stop_words=STOP_WORDS)
    tokens_freq = count_vectorizer.fit_transform(data["processed"])
    x = sparse.hstack((references_matrix.tocsr(), tokens_freq))

    tfidf = TfidfTransformer(norm="l2").fit_transform(x)
    if retur_vocab:
        return tfidf, list(references) + list(count_vectorizer.vocabulary_.keys()), len(references)
    else:
        return tfidf


def k_means_topic_detection(data, n_topics):
    tfidf = get_combined_tf_idf_matrix(data)
    model = KMeans(n_clusters=n_topics, n_jobs=-1)
    return model.fit_predict(tfidf)


def agglomerative_clustering_topic_detection(data, n_topics):
    tfidf = get_combined_tf_idf_matrix(data)
    x = TruncatedSVD(n_components=n_topics, algorithm="arpack").fit_transform(tfidf)
    model = AgglomerativeClustering(n_clusters=n_topics, affinity="cosine", linkage="complete", memory="./cache/")
    return model.fit_predict(x)


def lsa_topic_detection(data, n_topics, print_topic_tokens=False):
    tfidf, vocab, n_references = get_combined_tf_idf_matrix(data, retur_vocab=True)

    model = TruncatedSVD(n_components=n_topics if n_topics is not None else data.shape[0] // 200)
    topics = model.fit_transform(tfidf)
    n = n_topics if n_topics is not None else \
        np.sum(np.cumsum(model.explained_variance_ratio_) < EXPLAINED_VARIANCE_CUMSUM_THRESHOLD)

    result = topics[:, :n].argmax(axis=1)

    if print_topic_tokens:
        representative_tokens = model.components_[:, :n_references].argsort(axis=1)[:, -3:][:, ::-1]
        print("Most representative references for detected topics:")
        for topic_top_tokens in representative_tokens:
            print(', '.join([vocab[index] for index in topic_top_tokens]))

    return result


def baseline_solution(data, n_topics):
    mask = data.references.apply(lambda r: len(r) > 0)
    data["label"] = None

    data_ref = data[mask]
    references_count = defaultdict(int)
    for index, tweet in data_ref.iterrows():
        for ref in tweet.references:
            references_count[ref] += 1
    references_count = dict(references_count)
    labels = data_ref.references.apply(lambda references: max(references, key=lambda r: references_count[r]))
    data["label"][mask] = labels

    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, norm="l2")
    vectorizer.fit(data["processed"])
    y_labeled = data[mask]["label"].astype(str)
    x_labeled = vectorizer.transform(data[mask]["processed"])
    x_unlabeled = vectorizer.transform(data[~mask]["processed"])

    knn = KNeighborsClassifier(metric="cosine")
    knn.fit(x_labeled, y_labeled)

    data["label"][~mask] = knn.predict(x_unlabeled)
    return data["label"]
