import numpy                   as np
import matplotlib.pyplot       as plt
import scipy.cluster.hierarchy as scipy

from .tools import normalize_linkage


def dendrogram(Ls, n, l, format_labels=None, ax=None):
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

    outputs = []

    for i, L in enumerate(Ls):
        linkage = normalize_linkage(L)

        indices = L[:, :2].copy   ()
        flat    = indices .flatten()

        x_min  = L[:, 2].min() - 1
        x_max  = L[:, 2].max() + 1

        labels = [
            label(i)
            for i in np.sort(flat[flat < n])
        ]

        if ax is None:
            fig, current_ax = plt.subplots()
        elif isinstance(ax, (list, np.ndarray)):
            current_ax = ax[i]
            fig        = current_ax.figure
        else:
            current_ax = ax
            fig        = current_ax.figure

        scipy.dendrogram(linkage,
                         labels     =labels    ,
                         orientation='right'   ,
                         ax         =current_ax)

        current_ax.set_xlim((x_min, x_max))

        xticks = current_ax.get_xticks()

        current_ax.set_xticks     (xticks    )
        current_ax.set_xticklabels(l - xticks)

        outputs.append((fig, current_ax))

    return outputs
