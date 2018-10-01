from scipy.sparse import csr_matrix, lil_matrix
from itertools import combinations, permutations

from consts import NAMESPACE_SEP
from functools import reduce



def to_csr_matrix(ind_lists):
    """
    :param ind_lists: a list of pointers from each synsets to its relations
    :returns: CSR matrix representation
    """
    data = []
    indices = []
    indptr = []
    for i,l in enumerate(ind_lists):
        indptr.append(len(data))
        indices.extend(l)
        data.extend([1] * len(l))
    i += 1
    indptr.append(len(data))
    return csr_matrix((data,indices,indptr), shape=(i, i))


def join_sets(mat_dicts):
    """
    pre-condition: mat_dicts all have the same keys, and values all have the same dimensions
    semantic pre-condition: values nodes are same across matrix sets
    """
    ret = mat_dicts[0]
    for key in mat_dicts[0]:
        for d in mat_dicts[1:]:
            ret[key] += d[key]
    return ret


def tupstr(tup):
    return tup[0]+'\t'+','.join([str(t) for t in tup[1:]])


def canonicalize_name(str):
    return str.replace(' ', '_').lower()


def csr_eye(N):
    """
    :returns: sparse identity matrix of size n*n
    """
    return csr_matrix(([1] * N, (range(N), range(N))), [N, N])


def cyclic_triads(n):
    """
    :param n: members in set
    :returns: all triplets of members from range(n), order-agnostic, repetitions allowed
    """
    for i in range(n):
        yield (i,i,i)
        for j in range(i+1,n):
            yield (i,i,j)
            yield (i,j,j)
            for k in range(j+1, n):
                yield (i,j,k)
                yield (i,k,j)


def targets(g, idx):
    """
    :returns: list of target node ids for idx in graph g
    """
    return g[idx].nonzero()[1].tolist()


def sources(g, idx):
    """
    :returns: list of source node ids for idx in graph g
    """
    return g[:,idx].nonzero()[0].tolist()


def edges(g):
    """
    :returns: list of edge coordinates in tuple form
    """
    nz = g.nonzero()
    return list(zip(nz[0], nz[1]))


def find_key(cd, tup):
    """
    :param cd: cache dictionary where tuples may have unique form
    :param tup: one form of tuple
    """
    for tup_i in permutations(tup):
        if tup_i in cd:
            return tup_i
    return None


def find_cyclic_key(cd, tup):
    """
    :param cd: cache dictionary where tuples may have unique form up to cyclicity
    :param tup: one form of tuple
    """
    for i in range(len(tup)):
        tup_i = tup[i:] + tup[:i]
        if tup_i in cd:
            return tup_i
    return None


def co_graph(orig_g, direction='out'):
    """
    creates a graph featuring edges between all nodes that share a second-order relation
    :param orig_g: relation graph for original
    :param direction: if "out", means nodes share target, otherwise they share source
    """
    N = orig_g.shape[0]
    co_es = lil_matrix((N,N))
    for i in range(N):
        cliq = sources(orig_g, i) if direction == 'out' else targets(orig_g, i)
        for n1, n2 in zip(cliq, cliq):
            if n1 != n2:
                co_es[n1, n2] = 1
    return co_es.tocsr()

### UNUSED BUT COULD BE GENERALLY USEFUL ###

def joint_graph(graphs, aggregate=False):
    """
    :param graphs: list of graphs in CSR form.
    :param aggregate: edge weights are total from all graphs (instead of 1)
    :return: new CSR graph which is the 'or' sum of 'graphs'
    """
    agg_sum = reduce(lambda x,y: x+y, graphs)
    if aggregate:
        return agg_sum
    return agg_sum.sign()


def joint_n_graphs(graphs, r, aggregate=False):
    """
    :param graphs: list of graphs in CSR form
    :param r: number of graphs joined for each returned value (must be <= len(graphs))
    :param aggregate: edge weights are total from all graphs (instead of 1)
    :return: tuples of joint graphs and their indices in the input
    """
    for c in combinations(range(len(graphs)), r):
        yield joint_graph([graphs[i] for i in c], aggregate), c


def singleton_feature_map(graph_dict, func, namespace=''):
    """
    :param graph_dict: dictionary of graph name -> graph (CSR)
    :param func: feature function that accepts graph and returns single value
    :param namespace: (optional) prefix for feature name (underscore delimited)
    :return: dictionary from graph name (possibly prefixed) to value
    """
    if len(namespace) > 0:
        namespace += NAMESPACE_SEP
    return {(namespace+n):func(g) for n,g in list(graph_dict.items())}
