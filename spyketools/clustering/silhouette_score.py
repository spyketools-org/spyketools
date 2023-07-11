from sklearn.metrics import silhouette_score as sil

def silhouette_score(emb, labels_true):
    """
    Computation of embedding performance via Silhouette score[^1][^2] (Scikit Learn[^2]).

    Parameters
    ----------
    emb : numpy.array
        Embedding (2xM-dimesional array).
    labels_true : numpy.array
        Array with ground-truth labels.
    
    Returns
    -------
    score : float
        Silhouette score.

    **Examples:**

    ```python    
    import numpy as np

    # simulating data from three Gaussian distributions.
    points1 = np.random.normal(0, 10, (200,2))
    points2 = np.random.normal(30, 10, (200,2))
    points3 = np.random.normal(60, 10, (200,2))
    emb = np.concatenate([points1, points2, points3])
    labels = np.array([(i//200) for i in range(600)])
    
    # evaluation of clustering performance
    from spyketools.clustering import silhouette_score
    silhouette(emb, labels, n_components=None, random_state=0)
    # Output: 0.7772247810881237
    ```
    
    [^1]:
        *Peter J. Rousseeuw (1987). “Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis”. Computational and Applied Mathematics 20: 53-65.*
    [^2]:
        *Scikit Learn API. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score.*
    """

    return sil(emb, labels_true)