import temporal_network as tn
from make_temporal_network import _REF_DATE
import networkx as nx
import igraph as ig
from matplotlib import pyplot as plt

# just a record of thigns done

def _main(graph="alfred_long_cleaned.graphml"):
    NX_TN = tn.TemporalNetwork.read_graphml(graph)
    TN = ig.Graph.from_networkx(NX_TN)

    # clear networkx stuff
    NX_TN.clear()
    del NX_TN

    # compute pagerank
    pagerank = TN.pagerank(weight='weight')

    # infomap communities
    communities = TN.community_infomap(edge_weights='weight')

    # compute centrality measures
    harmonic_centrality = TN.harmonic_centrality(weight='weight')