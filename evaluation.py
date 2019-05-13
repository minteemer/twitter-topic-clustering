import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE

import preprocessing
import topic_detection


def evaluate(topic_detection_method, test_n_topics):
    results = []
    for n in test_n_topics:
        data = preprocessing.get_preprocessed_topics(n_topics=n)

        true_topics = data["topic"]
        x = data.drop(columns=["tweet", "topic"])

        if n is None:
            n = len(set(true_topics))

        predictions = topic_detection_method(x, n)
        ars = metrics.adjusted_rand_score(true_topics, predictions)
        results.append(ars)
        print(f"Adjusted Rand Score for {n} topics: {ars}")

    return results


def plot_dataset():
    data = preprocessing.get_preprocessed_topics(n_topics=10)
    tfidf = topic_detection.get_combined_tf_idf_matrix(data)
    model = TSNE(n_components=2, metric="cosine", verbose=1, n_iter=300)
    topics = model.fit_transform(tfidf.todense())

    color_map = {t: i for i, t in enumerate(data["topic"].unique())}
    colors = data["topic"].apply(lambda t: color_map[t])

    plt.scatter(topics[:, 0], topics[:, 1], c=colors, cmap=plt.get_cmap("hsv"))
    plt.show()


def test_lsa_n_topics_detection():
    # Test n_topics prediction
    for n in [10, 25, 50, 75, 100]:
        data = preprocessing.get_preprocessed_topics(n_topics=n)
        x = data.drop(columns=["tweet", "topic"])
        predictions = topic_detection.lsa_topic_detection(x, n_topics=None)

        print(f"Predicted number of topics by LSA: {len(set(predictions))}, actual number of topics: {n}")


def test_lsa_topics_representative_tokens_detection():
    # Test n_topics prediction
    n = 10
    data = preprocessing.get_preprocessed_topics(n_topics=n)
    x = data.drop(columns=["tweet", "topic"])
    topic_detection.lsa_topic_detection(x, n, print_topic_tokens=True)

    print("\nActual topic names:")
    for topic in data["topic"].unique():
        print(topic)


def evaluate_all_models():
    n_topics = [10, 20, 30, 40]

    print("Baseline solution: ")
    baseline = evaluate(topic_detection.baseline_solution, n_topics)

    print("K-means clustering: ")
    k_means = evaluate(topic_detection.k_means_topic_detection, n_topics)

    print("Agglomerative clustering: ")
    agglomerative = evaluate(topic_detection.agglomerative_clustering_topic_detection, n_topics)

    print("LSA topic detection: ")
    lsa = evaluate(topic_detection.lsa_topic_detection, n_topics)

    plt.plot(n_topics, baseline, label="Baseline")
    plt.plot(n_topics, k_means, label="K-means clustering")
    plt.plot(n_topics, agglomerative, label="Agglomerative clustering")
    plt.plot(n_topics, lsa, label="LSA")

    plt.title("Adjusted Rand Index score")
    plt.xlabel('Number of topics')
    plt.ylabel('ARI')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_dataset()
    # test_lsa_n_topics_detection()
    # test_lsa_topics_representative_tokens_detection()
    # evaluate_all_models()
