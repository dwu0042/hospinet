from temporal_network import TemporalNetwork
import networkx as nx
from functools import wraps
from typing import Callable
import warnings

def snapshots_outbound(G: TemporalNetwork):
    """Iterator of temporal snapshots of outbound edges that have a common source node time
    
    The snapshots will lose temporal information.
    The iterated elements will be a tuple of the time of the snapshot and the snapshot DiGraph"""
    for t, locs in G.snapshots.items():
        S = nx.DiGraph()
        for loc in locs:
            for neighbour, _t in G[loc, t]:
                existing_weight = S.edges.get((loc, neighbour), {'weight': 0})['weight']
                new_weight = existing_weight + G.edges[(loc, t), (neighbour, _t)]['weight']
                S.add_edge(loc, neighbour, weight=new_weight)
        yield t, S

def global_reaching_timeseries(G: TemporalNetwork, weight='weight'):
    """The timeseries of the global reaching centrality of the outbound snapshots of the temporal network
    
    The global reaching centrality ranges from 0 to 1, and is defined with respect to the local reaching centrality.
    The local reaching centrality, C(i), of a node, i, is the fraction of nodes that are reachable from that node.
    The global reaching centrality, C, is:
        $$ \frac {\sum (\max C(i) - C(i))} {N - 1} $$
    The global reaching centrality attains its maximum value of 1 for a star node: when there is a single node that is 
    connected all other nodes, and the other nodes are not conencted to each other.
    The local reaching centrality generalises for nodes of weighted graphs, where it is the average weight along the
    directed path from the node to another reachable node, where the directed path is chosen to maximise this average weight.

    This implementation depends on the implementation from networkx
    """
    @wraps(nx.global_reaching_centrality)
    def global_reaching_centrality(G, *args, **kwargs):
        if G.number_of_nodes() < 1:
            return 0
        return nx.global_reaching_centrality(G, *args, **kwargs)

    return sorted((t, global_reaching_centrality(S, weight=weight)) for t, S in snapshots_outbound(G))

def failover(func: Callable, default=0):
    """Decorates a function, returning a failover default value if it raises an Exception of some kind"""
    @wraps(func)
    def failovered_func(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception as e:
            warnings.warn(str(e), RuntimeWarning)
            ret = default
        finally:
            return ret
    return failovered_func

def temporal_timeseries(G: TemporalNetwork, method: Callable, safe=False, default=0, *args, **kwargs):
    """Evaluates a network metric over the outbound snapshots of a temporal network
    
    Parameters
    ----------
    G: temporal network of the 
    method: network metric to evaluate (not necessarily scalar)
    safe: If False, overrides the returned value if the method raises an Exception
    default: Replacement value if safe is False

    Returns
    -------
    A list consisting of tuple elements (t, metric), sorted by t (ascending order)
    """
    if not safe:
        method = failover(method, default=default)
    
    return sorted((t, method(S, *args, **kwargs)) for t, S in snapshots_outbound(G))