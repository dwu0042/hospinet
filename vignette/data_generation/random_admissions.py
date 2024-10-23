import datetime
import random
import polars as pl
from string import ascii_uppercase

from typing import Sequence


def _seed_with(seed=None):
    random.seed(seed)


START_DATE = datetime.datetime(year=2020, month=4, day=1)
END_DATE = datetime.datetime(year=2021, month=6, day=28)


def random_date_interval(
    start=START_DATE,
    end=END_DATE,
) -> Sequence[datetime.datetime]:
    return sorted(random.uniform(start, end) for _ in range(2))


def random_entry(
    patients: int = 10,
    hospitals: int = 5,
    date_start=START_DATE,
    date_end=END_DATE,
):
    date_interval = random_date_interval(start=date_start, end=date_end)
    return {
        "patient": random.randint(1, patients),
        "hospital": random.choice(
            ascii_uppercase[:hospitals]
        ),  # sucks to suck if hospitals > 26
        "admission": date_interval[0],
        "discharge": date_interval[1],
    }


def generate_admissions(n_entries: int = 100, **kwargs) -> pl.DataFrame:
    return pl.from_dicts([random_entry(**kwargs) for _ in range(n_entries)]).sort(
        "admission", "discharge"
    )


def main(seed=None, **kwargs):
    _seed_with(seed)
    df = generate_admissions(**kwargs)
    df.write_csv("vignette/data/admissions.csv")


if __name__ == "__main__":
    main(seed=0x12B342B71, patients=4)
