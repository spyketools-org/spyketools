import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def ari_KMeans(emb, labels_true, n_components=None, random_state=None):
    """
    Adjusted Rand Index[^1][^2] (ARI) score via KMeans clustering (Scikit Learn[^3]).

    Parameters
    ----------
    emb : numpy.array
        Embedding (2xM-dimesional array).
    labels_true : numpy.array
        Array with ground-truth labels.
    n_components : int
        Number of components to find using KMeans algorithm.
    random_state : int
        Seed for random_state param (Scikit Learn).
    
    Returns
    -------
    score : float
        ARI score value using KMeans clustering.

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
    from spyketools.clustering import ari_KMeans
    ari_KMeans(emb, labels, n_components=None, random_state=0)
    # Output: 0.7772247810881237
    ```
    
    [^1]:
        *L. Hubert and P. Arabie, Comparing Partitions, Journal of Classification 1985 https://link.springer.com/article/10.1007%2FBF01908075*
    [^2]:
        *D. Steinley, Properties of the Hubert-Arabie adjusted Rand index, Psychological Methods 2004.*
    [^3]:
        *Scikit Learn API. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html.*

    """

    if n_components is None:
        n_components = len(np.unique(labels_true))
    labels_pred = KMeans(n_clusters=n_components, random_state=random_state).fit_predict(emb)#.labels_
    return adjusted_rand_score(labels_true, labels_pred)


def ari_GM(emb, labels_true, max_iter=100, n_components=None, random_state=None):
    """
    Adjusted Rand Index[^1][^2] (ARI) score via Gaussian Mixture (GM) clustering (Scikit Learn[^3]).

    Parameters
    ----------
    emb : numpy.array
        Embedding (2xM-dimesional array).
    labels_true : numpy.array
        Array with ground-truth labels.
    max_iter : int
        Maximum number of iteration for GM algorithm.
    n_components : int
        Number of components to find using GM algorithm.
    random_state : int
        Seed for random_state param (Scikit Learn).

    Returns
    -------
    score : float
        ARI score value using GM clustering.

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
    from spyketools.clustering import ari_GM

    ari_GM(emb, labels, n_components=None, random_state=0)
    
    # Output: 0.7928597274827337
    ```
    
    [^1]:
        *L. Hubert and P. Arabie, Comparing Partitions, Journal of Classification 1985 https://link.springer.com/article/10.1007%2FBF01908075*
    [^2]:
        *D. Steinley, Properties of the Hubert-Arabie adjusted Rand index, Psychological Methods 2004.*
    [^3]:
        *Scikit Learn API. https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.*
    """

    if n_components is None:
        n_components = len(np.unique(labels_true))
    labels_pred = GaussianMixture(n_components=n_components, max_iter=max_iter, random_state=random_state).fit_predict(emb)
    return adjusted_rand_score(labels_true, labels_pred)
    
def ari_HDBSCAN(emb, labels_true):
    """
    Adjusted Rand Index[^1][^2] (ARI) score via "Hierarchical Density-Based Spatial 
    Clustering of Applications with Noise" (HDBSCAN) algorithm[^3][^4].

    Parameters
    ----------
    emb : numpy.array
        Embedding (2xM-dimesional array).
    labels_true : numpy.array
        Array with ground-truth labels.

    Returns
    -------
    score : float
        ARI score value using HDBSCAN clustering.

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
    from spyketools.clustering import ari_HDBSCAN
    ari_HDBSCAN(emb, labels, n_components=None, random_state=0)
    # Output: 0.7656426881931003
    ```
    
    [^1]:
        *L. Hubert and P. Arabie, Comparing Partitions, Journal of Classification 1985 https://link.springer.com/article/10.1007%2FBF01908075*
    [^2]:
        *D. Steinley, Properties of the Hubert-Arabie adjusted Rand index, Psychological Methods 2004.*
    [^3]:
        *McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. J. Open Source Softw., 2(11), 205*.
    [^4]:
        *HDBSCAN  docs. https://hdbscan.readthedocs.io/en/latest/index.html.*
    """
    labels_pred = hdbscan.HDBSCAN().fit(emb).labels_
    return adjusted_rand_score(labels_true, labels_pred)

def ari_custom(labels_true, labels_pred):
    """
    Adjusted Rand Index[^1][^2] (ARI) score using a custom clustering algorithm.

    Parameters
    ----------
    labels_true : numpy.array
        Array with ground-truth labels.
    labels_pred : numpy.array
        Array with labels from any other clustering algorithm.

    Returns
    -------
    score : float
        ARI score value using a custom clustering algorithm.

    **Examples:**
    
    [^1]:
        *L. Hubert and P. Arabie, Comparing Partitions, Journal of Classification 1985 https://link.springer.com/article/10.1007%2FBF01908075*
    [^2]:
        *D. Steinley, Properties of the Hubert-Arabie adjusted Rand index, Psychological Methods 2004.*
    """
    return adjusted_rand_score(labels_true, labels_pred)