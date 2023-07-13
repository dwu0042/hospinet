from temporal_network import TemporalNetwork
import polars as pl
import datetime

_REF_DATE = datetime.datetime(year=2017, month=3, day=1)

def normalise_dates(df: pl.DataFrame, cols, ref_date=_REF_DATE):
    """Normalises Datetime columns to the number of days past a given reference date"""
    return df.with_columns(*(
        (pl.col(col) - ref_date).dt.seconds() /60/60/24
        for col in cols
    )
    )

_empty_edge = {'weight': 0}
def convert_presence_to_network(presence: pl.DataFrame, discretisation=1, return_window=365):
    """Converts a Dataframe of presences to a temporal network with base units of days
    
    Parameters
    ----------
    presence: pola.rs DataFrame of presence. Assumes that the columns are ['sID', 'fID', 'Adate', 'Ddate']
              Assumes that 'Adate' and 'Ddate' columns are normalised to integers
    discretisation: time discretisation of the temporal network
    return_window: threshold over which successive presences are ignored

    Returns
    -------
    TemporalNetwork where edges represent patients that have transferred between given locations.
    """
    G = TemporalNetwork()

    presence = (presence.sort(pl.col('Adate'))
                        .with_columns(
                            pl.arange(
                                pl.col('Adate').floordiv(discretisation)*discretisation, 
                                pl.col('Ddate')+1, 
                                discretisation)
                            .alias('present'))
                        .explode('present')
    )

    for partition in presence.partition_by('sID'):
        partition = partition.sort('present', 'Adate').select('fID', 'present')
        for row0, row1 in zip(partition.shift(1).iter_rows(), partition.iter_rows()):
            if row0[-1] is not None and ((row1[-1] - row0[-1]) < return_window):
                nodes = (row0, row1)
                w = G.edges.get(nodes, _empty_edge)['weight'] + 1
                G.add_edge(*nodes, weight=w)

    return G
