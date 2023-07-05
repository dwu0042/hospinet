import numpy as np
rng = np.random.default_rng()
import polars as pl
import datetime
import string
from itertools import cycle

def make_journey(t0=0, duration=10, swap_period=6, prob_leaving=0.7, locations=12):
    t = t0
    t1 = None
    locs = []
    all_locs = set(map(''.join, zip(cycle(string.ascii_uppercase), map(str, range(locations)))))
    locs.append((t, rng.choice(list(all_locs))))
    while t < (t0+duration):
        dt = rng.exponential(swap_period)
        t = t + dt
        leave = rng.random() < prob_leaving
        if leave:
            t1 = t
            break
        else:
            next_loc = rng.choice(list(all_locs - {locs[-1][-1]}))
            locs.append((t, next_loc))
    if t1 is None:
        t1 = t + rng.exponential(swap_period)
    locs.append((t1, None))
    return locs

def translate_journey(journey, sID=None):
    rows = []
    for (t0, loc0), (t1, loc1) in zip(journey[:-1], journey[1:]):
        rows.append((sID, loc0, t0, t1))
    return rows

_REF_DATE = datetime.datetime(year=2017, month=3, day=1)
def generate_presences(n_patients=100, n_locations=20, duration=10, ref_date=_REF_DATE):
    n_patients = 100
    n_locations = 20
    journeys = [
        make_journey(
            t0 = rng.uniform(0, n_patients),
            duration = duration,
            locations = n_locations,
        ) for _ in range(n_patients)
    ]
    presences = [entry 
                for patient, journey in enumerate(journeys)
                for entry in translate_journey(journey, patient)
                ]
    df = pl.from_records(presences,
                        schema=['sID', 'fID', 'Adate', 'Ddate']
                        )

    df = df.with_columns(
        (pl.duration(days=pl.col('Adate'), 
                     seconds=pl.col('Adate').mod(1)*60*60*24) 
            + _REF_DATE).alias('Adate').cast(pl.Datetime),
        (pl.duration(days=pl.col('Ddate'),
                     seconds=pl.col('Ddate').mod(1)*60*60*24) 
            + _REF_DATE).alias('Ddate').cast(pl.Datetime),
    )

    return df