import networkx as nx
from collections import defaultdict

class TemporalNetwork(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

        self.snapshots = defaultdict(set)
        self.present = defaultdict(set)
        # self.locations = set()

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        _super_ret = super().add_edge(u_of_edge, v_of_edge, **attr)

        for loc, t in (u_of_edge, v_of_edge):
            self.snapshots[t].add(loc)
            self.present[loc].add(t)

        return _super_ret

    def nodes_at_time(self, t):
        return [(loc, t) for loc in self.snapshots[t]]
    
    def when_present(self, loc):
        return [(loc, t) for t in self.present[loc]]