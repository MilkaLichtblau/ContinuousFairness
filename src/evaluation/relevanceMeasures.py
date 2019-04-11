import numpy as np
import matplotlib.pyplot as plt


def pak(k, ranking, originalRanking):

    """
    Calculates P@k

    @param k: truncation point/length of the ranking
    @param ranking: list of candidates selected for the ranking

    return value for P@k
    """

    # cut the ranking at the given truncation point k
    pakRanking = ranking[:k]
    pakOrig = originalRanking[:k]

    plt.plot(ranking)
    plt.plot(originalRanking)
    plt.show()

    # check if docIDs at current position occur in original and in new ranking
    pak = len([docId for docId in pakRanking if docId in pakOrig])

    # discount with truncation point
    pak = pak / k

    return pak


def ap(ranking, originalRanking):

    """
    Calculate AP

    @param ranking: list of candidates selected for the ranking
    @param originalRanking {ndarray}: original positions

    return AP
    """

    # initialize AP
    ap = 0

    # check if rank in current ranking equals position in color-blind
    for k in range(len(ranking)):
        ap += pak(k, ranking, originalRanking)

    return ap / len(ranking)


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

