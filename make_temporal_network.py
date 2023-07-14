from temporal_network import TemporalNetwork, EMPTY_EDGE as _empty_edge
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
    ).sort('sID', 'present', 'Adate')

    G.add_nodes_from(tuple(x) for x in presence.select('fID', 'present').unique().iter_rows())

    edges = (presence
             # get the previous record
            .with_columns(
                pl.col('sID', 'present', 'fID').shift(1).map_alias(lambda x: f"prev_{x}"),
            )
            # check same individual, and within the return window
            .filter(
                (pl.col('sID').eq(pl.col('prev_sID')))
                & ((pl.col('present') - pl.col('prev_present')) < return_window)
            )
            # pull columns
            .select('prev_fID', 'prev_present', 'fID', 'present')
            # get counts of edges
            .groupby('*').count()
    )

    G.add_weighted_edges_from(((ux, ut), (vx, vt), w) for ux, ut, vx, vt, w in edges.iter_rows())

    G.snapshots = dict(presence.groupby('present').all().select(pl.col('present'), pl.col('fID').list.unique()).to_numpy())

    G.present = dict(presence.groupby('fID').all().select(pl.col('fID'), pl.col('present').list.unique()).to_numpy())

    return G
