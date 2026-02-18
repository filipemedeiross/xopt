import numpy                   as np
import matplotlib.pyplot       as plt
import scipy.cluster.hierarchy as scipy

from .tools import normalize_linkage


def dendrogram(L, n, l, format_labels=None):
    '''
    Recebe  -> lista de linkages e os valores de N e L para a instância
    Retorna -> lista de objetos Figure, cada qual representando um cluster
    '''
    if isinstance(L, np.ndarray):
        L = [L]

    if format_labels:
        label = lambda x: format_labels[int(x)]
    else:
        label = lambda x: f'X{int(x)}'

    figs = []
    for c in range(len(L)):
        indices = L[c][:, :2].flatten()
        labels  = [label(i) for i in np.sort(indices[np.where(indices < n)])]

        x_min = L[c][:, 2].min() - 1
        x_max = L[c][:, 2].max() + 1

        fig, _ = plt.subplots()
        scipy.dendrogram(normalize_linkage(L[c]),
                         labels=labels,
                         orientation='right')
        plt.xlim((x_min, x_max))
        plt.xticks(plt.xticks()[0], l - plt.xticks()[0])

        figs.append(fig)

    return figs
