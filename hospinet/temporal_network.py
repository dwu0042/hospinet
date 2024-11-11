"""This submodule defines the TemporalNetwork class that represents a flattened time-discretised version of patient transfers"""

import polars as pl
import networkx as nx

from collections import defaultdict

from typing import SupportsFloat, Hashable, Self, Sequence, Iterable, Set, Any
from os import PathLike


EMPTY_EDGE = {"weight": 0}


class TemporalNetwork(nx.DiGraph):
    """A temporal network for hospital patient transfers"""

    def __init__(self, incoming_graph_data=None, **attr) -> None:
        """Create a temporal network that is based on a `networkx.DiGraph`

        Adds the internal snapshots and present lookup dictionaries to the base construction

        Args:
            incoming_graph_data (optional): Graph data to parse by networkx
            **attr: Graph attributes to pass to networkx constructor
        """
        super().__init__(incoming_graph_data, **attr)

        self.snapshots = defaultdict(set)
        self.present = defaultdict(set)
        # self.locations = set()

    def add_edge(self, u_of_edge, v_of_edge, **attr) -> Any:
        """Add an edge to the temporal network

        Updates the internal presence dictionaries
        Also see `networkx.DiGraph.add_edge`

        Args:
            u_of_edge: name of the source node of the edge to add
            v_of_edge: name of the target node of the edge to add
            **attr: edge attributes to add
        """
        _super_ret = super().add_edge(u_of_edge, v_of_edge, **attr)

        for loc, t in (u_of_edge, v_of_edge):
            self.snapshots[t].add(loc)
            self.present[loc].add(t)

        return _super_ret

    def locs_at_time(self, t: Hashable) -> Set[Hashable]:
        """Returns a set of all locations present at a given time

        Args:
            t (Hashable): time at which to slice nodes
        """
        return self.snapshots[t]

    def nodes_at_time(self, t: Hashable) -> Sequence[Hashable]:
        """Returns a list of nodes for all locations present at a given time

        Args:
            t (Hashable): time at which to slice nodes
        """

        return self.nodes_like(self.locs_at_time(t), t)

    def times_for_place(self, loc: Hashable) -> Set[Hashable]:
        """Returns a set of all times a location is present at

        Args:
            loc (Hashable): location to look up
        """
        return self.present[loc]

    def nodes_for_place(self, loc: Hashable) -> Sequence[Hashable]:
        """Returns a list of nodes for all times a location is present at

        Args:
            loc (Hashable): location to look up
        """
        return self.nodes_like(loc, self.times_for_place(loc))

    @staticmethod
    def nodes_like(
        loc: Hashable | Iterable[Hashable], t: Hashable | Iterable[Hashable]
    ):
        """Combines loc and t inputs into node lookups

        Args:
            loc (Hashable | Iterable[Hashable]): Either a single location, or a list of locations
            t (Hashable | Iterable[Hashable]): Either a single time, or a list of times

        Notes:
            A maximum of one input argument can be a list

        Returns:
            Sequence[Hashable]: A list of tuples that can be looked up as nodes
        """
        # Treat strings as singletons
        is_loc_iterable = isinstance(loc, Iterable) and not isinstance(loc, str)
        is_t_iterable = isinstance(t, Iterable) and not isinstance(t, str)

        if not is_loc_iterable and not is_t_iterable:
            return [(loc, t)]
        elif is_loc_iterable and not is_t_iterable:
            return [(item, t) for item in loc]
        elif not is_loc_iterable and is_t_iterable:
            return [(loc, item) for item in t]
        else:
            raise ValueError(
                "Unexpected input types. Only one input can be an iterable at a time."
            )

    @classmethod
    def from_timenode_projection(cls, G: nx.DiGraph) -> Self:
        """Construct a temporal graph where the nodes are tuples of form (loc, t)"""
        TN = cls(G)
        for (u_loc, u_t), (v_loc, v_t) in G.edges:
            TN.snapshots[u_t].add(u_loc)
            TN.snapshots[v_t].add(v_loc)
            TN.present[u_loc].add(u_t)
            TN.present[v_loc].add(v_t)
        return TN

    @classmethod
    def read_graphml(cls, path: str | PathLike, *args, **kwargs) -> Self:
        f"""Constructs a {cls} graph from a given graphml file"""

        def parse_tuple(tuple_str):
            loc, t = tuple_str.lstrip("(").rstrip(")").split(",")
            loc = loc.strip("'")
            return loc, int(t)

        G = nx.read_graphml(path, node_type=parse_tuple, *args, **kwargs)

        return cls.from_timenode_projection(G)

    def to_static(self) -> nx.DiGraph:
        """Projects a TemporalNetwork to a static DiGraph on its locations"""
        S = nx.DiGraph()
        Nt = len(self.snapshots)
        for loc, ts in self.present.items():
            S.add_node(loc, present=len(ts) / Nt)
            for t in ts:
                for (nbr_loc, nbr_t), weight in self[loc, t].items():
                    if loc != nbr_loc:
                        existing_weight = S.get_edge_data(loc, nbr_loc, EMPTY_EDGE)[
                            "weight"
                        ]
                        new_weight = existing_weight + weight["weight"]
                        S.add_edge(loc, nbr_loc, weight=new_weight)
        return S

    @classmethod
    def from_presence(
        cls,
        presence: pl.DataFrame,
        discretisation: int = 1,
        return_window: SupportsFloat = 365,
    ) -> Self:
        """Converts a Dataframe of presences to a temporal network with base units of days

        Args:
            presence (pl.DataFrame): dataframe of presence. Assumes that the columns are ['sID', 'fID', 'Adate', 'Ddate']
                Assumes that 'Adate' and 'Ddate' columns are normalised to integers
            discretisation (int, optional): time discretisation of the temporal network. Defaults to 1.
            return_window (SupportsFloat, optional): threshold over which successive presences are ignored. Defaults to 365

        Returns:
            TemporalNetwork where edges represent patients that have transferred between given locations.
        """
        G = cls()

        presence = (
            presence.sort(pl.col("Adate"))
            .with_columns(
                pl.int_ranges(
                    pl.col("Adate").floordiv(discretisation) * discretisation,
                    pl.col("Ddate") + 1,
                    discretisation,
                ).alias("present")
            )
            .explode("present")
        ).sort("sID", "present", "Adate")

        G.add_nodes_from(
            tuple(x) for x in presence.select("fID", "present").unique().iter_rows()
        )

        edges = (
            presence
            # get the previous record
            .with_columns(
                pl.col("sID", "present", "fID")
                .shift(1)
                .name.map(lambda x: f"prev_{x}"),
            )
            # check same individual, and within the return window
            .filter(
                (pl.col("sID").eq(pl.col("prev_sID")))
                & ((pl.col("present") - pl.col("prev_present")) < return_window)
            )
            # pull columns
            .select("prev_fID", "prev_present", "fID", "present")
            # get counts of edges
            .group_by("*").len()
        )

        G.add_weighted_edges_from(
            ((ux, ut), (vx, vt), w) for ux, ut, vx, vt, w in edges.iter_rows()
        )

        G.snapshots = {
            k: set(v)
            for k, v in presence.group_by("present")
            .all()
            .select(pl.col("present"), pl.col("fID").list.unique())
            .to_numpy()
        }

        G.present = {
            k: set(v)
            for k, v in presence.group_by("fID")
            .all()
            .select(pl.col("fID"), pl.col("present").list.unique())
            .to_numpy()
        }

        return G

    def write_graphml(self, outfile: str | PathLike, *args, **kwargs) -> None:
        """Write the temporal network to a graphml format file

        Args:
            outfile (str | PathLike): Path to write file to
            *args: positional arguments to pass to `networkx.write_graphml`
            **kwargs: keyword arguments to pass to `networkx.write_graphml`
        """
        nx.write_graphml(self, outfile, *args, **kwargs)

    def write_lgl(self, outfile: str | PathLike, weight="weight") -> None:
        """Write the temporal network out to an lgl-style file

        Args:
            outfile (str | Pathlike): Path to write file to
            weight (str, optional): Edge attribute to use as weight column in lgl format. If this attribute does not exist for an edge, replaces with empty
        """

        # instead of using networkx.write_weighted_edgelist
        # we do this to customise the printing of the tuple-nodes:
        # they otherwise would have whitespace, which breaks the lgl format
        with open(outfile, "w") as fp:
            for (l_fr, t_fr), (l_to, t_to), attr_dict in nx.to_edgelist(self):
                fp.write(
                    f"""({l_fr},{t_fr}) ({l_to},{t_to}) {attr_dict.get(weight)}\n"""
                )
