from temporal_network import TemporalNetwork
import polars as pl
import numpy as np
import datetime

_REF_DATE = datetime.datetime(year=2017, month=3, day=1)

def normalise_dates(df: pl.DataFrame, cols, ref_date=_REF_DATE):
    return df.with_columns(*(
        (pl.col(col) - _REF_DATE).dt.seconds() /60/60/24
        for col in cols
    )
    )

_empty_edge = {'weight': 0}
def convert_presence_to_network(presence: pl.DataFrame, discretisation=1, return_window=365):

    G = TemporalNetwork()

    presence = (presence.sort(pl.col('Adate'))
                        .with_columns(pl.arange(pl.col('Adate'), pl.col('Ddate')+1).alias('present'))
                        .explode('present')
    )

    for partition in presence.partition_by('sID'):
        for row0, row1 in zip(partition.iter_rows(), partition.shift(1).iter_rows()):
            if row1[-1] is not None and row1[-1] - row0[-1] < return_window:
                nodes = ((row0[1], row0[-1]), (row1[1], row1[-1]))
                w = G.edges.get(nodes, _empty_edge)['weight'] + 1
                G.add_edge(*nodes, weight=w)

    return G
