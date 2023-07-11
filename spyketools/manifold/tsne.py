from sklearn.manifold import TSNE

def tsne_emb(diss_matrix, metric='precomputed', perplexity=30, random_state=None):
    """
    Computation of 2D T-distributed Stochastic Neighbor Embedding (t-SNE) [^1][^2][^3][^4] (Scikit Learn[^5]).

    Parameters
    ----------
    diss_matrix : numpy.array
        Dissimilarity matrix from pairwise distances (i.e. `(M,M)`-dimesional array).
    metric : str
        Metric to compute embedding. By default, the dissimilarity matrix `diss_matrix` is used (`'pre-computed'`).
    perplexity: float
        Perplexity value for t-SNE embedding.
    random_state: int
        Seed for random_state param (Scikit Learn).
    
    Returns
    -------
    emb : numpy.array
        2D t-SNE Embedding.

    [^1]: 
        *van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data. 
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.*
    [^2]:
        *van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding. https://lvdmaaten.github.io/tsne/.*
    [^3]:
        *L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014. 
        https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf*
    [^4]:
        *Belkina, A. C., Ciccolella, C. O., Anno, R., Halpert, R., Spidlen, J., & Snyder-Cappione, J. E. (2019). 
        Automated optimized parameters for T-distributed stochastic neighbor embedding improve visualization 
        and analysis of large datasets. Nature Communications, 10(1), 1-12.*
    [^5]:
        Scikit-learn docs. https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    
    emb = TSNE(
        metric='precomputed', 
        n_components=2, 
        random_state=random_state, 
        perplexity=perplexity).fit_transform(diss_matrix)
    return emb