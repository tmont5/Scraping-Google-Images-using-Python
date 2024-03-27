"""Methods to visualize the high-dimensional embeddings being produced by an embedding network. We use nonlinear
dimension reduction techniques (t-SNE, UMAP) as opposed to linear ones (PCA, ICA, etc.) since neural networks learn
highly nonlinear relationships."""

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Iterable
from umap import UMAP

def plot_tsne_reduced_embeddings(embeddings: np.array, labels: np.array, dimension: int = 2, classes_to_plot: Optional[Iterable] = None, show: bool = True):
    """Use t-SNE to reduce embeddings dimension and scatterplot them in desired number of dimensions.
    
    Args:
        embeddings (ndarray, (n, d)): Array of n d-dimensional embeddings to visualize
        labels (ndarray, (n,)): Array of class labels for each embedding (used in plot legend)
        dimension (int): Desired number of dimensions to project embeddings down to (must be 2 or 3)
        classes_to_plot (Optional[Iterable]): Allows specification of classes to display on scatterplot.
                                              If None, 4 are randomly chosen.
        show (bool): Specifies whether/not to call plt.show() after generating plot.

    Returns:
        projections (ndarray, (n, dimension)): Array of TSNE projections used in the scatterplot.
    """
    if dimension not in {2, 3}:
        raise ValueError("Dimension argument must range between 1 and 3.")

    projections = TSNE(n_components=dimension).fit_transform(embeddings)

    projections_are_2d = (dimension == 2)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Reduced-Dimensional Embeddings", fontsize=20)
    ax = fig.add_subplot() if projections_are_2d else fig.add_subplot(projection='3d')
    
    if classes_to_plot is None:
        classes_to_plot = sorted(np.random.choice(np.unique(labels), size=4, replace=False))

    if projections_are_2d:
        for c in classes_to_plot:
            ax.scatter(*projections[labels == c].T, label=str(c))
                
    else:
        for c in classes_to_plot:
                ax.scatter3D(*projections[labels == c].T, label=str(c))

    ax.legend()
    
    if show:
        plt.show()

    return projections
        
def plot_umap_reduced_embeddings(embeddings: np.array, labels: np.array, dimension: int = 2, classes_to_plot: Optional[Iterable] = None, show: bool = True):
    """Use UMAP to reduce embeddings dimension and scatterplot them in desired number of dimensions.
    
    Args:
        embeddings (ndarray, (n, d)): Array of n d-dimensional embeddings to visualize
        labels (ndarray, (n,)): Array of class labels for each embedding (used in plot legend)
        dimension (int): Desired number of dimensions to project embeddings down to (must be 2 or 3)
        classes_to_plot (Optional[Iterable]): Allows specification of classes to display on scatterplot.
                                              If None, 4 are randomly chosen.
        show (bool): Specifies whether/not to call plt.show() after generating plot.

    Returns:
        projections (ndarray, (n, dimension)): Array of TSNE projections used in the scatterplot.
    """
    if dimension not in {2, 3}:
        raise ValueError("Dimension argument must range between 1 and 3.")

    projections = UMAP(n_components=dimension).fit_transform(embeddings)

    projections_are_2d = (dimension == 2)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Reduced-Dimensional Embeddings", fontsize=20)
    ax = fig.add_subplot() if projections_are_2d else fig.add_subplot(projection='3d')
    
    if classes_to_plot is None:
        classes_to_plot = sorted(np.random.choice(np.unique(labels), size=4, replace=False))

    if projections_are_2d:
        for c in classes_to_plot:
            ax.scatter(*projections[labels == c].T, label=str(c))
                
    else:
        for c in classes_to_plot:
                ax.scatter3D(*projections[labels == c].T, label=str(c))

    ax.legend()
    
    if show:
        plt.show()

    return projections
        