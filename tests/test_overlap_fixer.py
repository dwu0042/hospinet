from hospinet import overlap_fixer as ovlfxr
import polars as pl
from datetime import datetime


def test_simple_overlap():
    df = pl.DataFrame(
        {
            "sID": [1, 1, 1, 1],
            "fID": ["A", "B", "C", "D"],
            "Adate": [
                datetime(year=2000, month=1, day=1),
                datetime(year=2000, month=2, day=1),
                datetime(year=2000, month=3, day=1),
                datetime(year=2000, month=4, day=1),
            ],
            "Ddate": [
                datetime(year=2000, month=1, day=15),
                datetime(year=2000, month=3, day=10),
                datetime(year=2000, month=3, day=28),
                datetime(year=2000, month=4, day=15),
            ],
        }
    )

    ff = ovlfxr.fix_overlaps(df, 1)

    assert ovlfxr.num_overlaps(ff) == 0


def test_consuming_overlap():
    df = pl.DataFrame(
        {
            "sID": [1, 1, 1, 1],
            "fID": ["A", "B", "C", "D"],
            "Adate": [
                datetime(year=2000, month=1, day=1),
                datetime(year=2000, month=2, day=1),
                datetime(year=2000, month=3, day=1),
                datetime(year=2000, month=4, day=1),
            ],
            "Ddate": [
                datetime(year=2000, month=1, day=15),
                datetime(year=2000, month=3, day=28),
                datetime(year=2000, month=3, day=10),
                datetime(year=2000, month=4, day=15),
            ],
        }
    )

    ff = ovlfxr.fix_overlaps(df, 1)

    assert ovlfxr.num_overlaps(ff) == 0


def test_dead_heat_overlap():
    df = pl.DataFrame(
        {
            "sID": [1, 1, 1, 1],
            "fID": ["A", "B", "C", "D"],
            "Adate": [
                datetime(year=2000, month=1, day=1),
                datetime(year=2000, month=2, day=1),
                datetime(year=2000, month=2, day=1),
                datetime(year=2000, month=4, day=1),
            ],
            "Ddate": [
                datetime(year=2000, month=1, day=15),
                datetime(year=2000, month=3, day=10),
                datetime(year=2000, month=3, day=28),
                datetime(year=2000, month=4, day=15),
            ],
        }
    )

    ff = ovlfxr.fix_overlaps(df, 1)

    assert ovlfxr.num_overlaps(ff) == 0


if __name__ == "__main__":
    test_simple_overlap()
    test_consuming_overlap()
    test_dead_heat_overlap()
