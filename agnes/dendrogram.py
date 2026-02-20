import numpy                   as np
import matplotlib.pyplot       as plt
import scipy.cluster.hierarchy as scipy

from .tools import normalize_linkage


def dendrogram(Ls, n, l, format_labels=None):
    '''
    Recebe  -> lista de linkages e os valores de N e L para a instância
    Retorna -> lista de objetos Figure, cada qual representando um cluster
    '''
    if isinstance(Ls, np.ndarray):
        Ls = [Ls]

    if format_labels:
        label = lambda x: format_labels[int(x)]
    else:
        label = lambda x: f'X{int(x)}'

    figs = []

    for L in Ls:
        linkage = normalize_linkage(L)

        indices = L[:, :2].copy   ()
        flat    = indices .flatten()

        x_min  = L[:, 2].min() - 1
        x_max  = L[:, 2].max() + 1

        labels = [
            label(i)
            for i in np.sort(flat[flat < n])
        ]

        fig, _ = plt.subplots()

        scipy.dendrogram(linkage,
                         labels     =labels ,
                         orientation='right')

        plt.xlim  ((x_min, x_max))
        plt.xticks(plt.xticks()[0], l - plt.xticks()[0])

        figs.append(fig)

    return figs
