from umap import UMAP

def umap_emb(diss_matrix, metric='precomputed', n_neighbors=15, min_dist=0.1, n_components=2, random_state=None):
    """
    Computation of Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP) [^1] (Docs. [^2]).

    Parameters
    ----------
    diss_matrix : numpy.array
        Dissimilarity matrix from pairwise distances (i.e. `(M,M)`-dimesional array).
    metric : str
        Metric to compute embedding. By default, the dissimilarity matrix `diss_matrix` is used (`'pre-computed'`).
    n_neighbors : int
        Size of local nighborhood for UMAP.
    min_dist : float
        Minimum distance apart that points are allowed to be in the low dimensional representation.
    n_components : int
        Dimensionality of the reduced dimension space of embedding.
    random_state: int
        Seed for random_state param.
    
    Returns
    -------
    emb : numpy.array
        2D UMAP embedding.

    [^1]: 
        *McInnes, L., Healy, J., & Melville, J. (2018). Umap: Uniform manifold approximation 
        and projection for dimension reduction. arXiv preprint arXiv:1802.03426.*
    [^2]:
        UMAP docs. https://umap-learn.readthedocs.io/en/latest/index.html.
    """
    
    emb = UMAP(
        metric='precomputed', 
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        n_components=n_components, 
        random_state=random_state).fit_transform(diss_matrix)
    return emb