import polars as pl
from time import perf_counter as tic


def scan_overlaps(df: pl.DataFrame):
    """Creates a LazyFrame of the overlaps present in teh dataframe"""
    overlaps = (
        df.lazy()
        # ensure sorted
        .sort("sID", "Adate", "Ddate")
        # map up next row
        .filter(
            pl.col("sID").eq(pl.col("sID").shift(-1))
            & pl.col("Adate").shift(-1).lt(pl.col("Ddate"))
        )
    )

    return overlaps


def num_overlaps(df: pl.DataFrame):
    return scan_overlaps(df).collect().height


def fix_overlaps_single_iter(df: pl.DataFrame):
    """Performs one iteration of overlap correction on successive presence events"""

    # expression to filter out non-overlaps (assigns nulls)
    reject_no_overlaps = (
        pl.when(pl.col("sID").ne(pl.col("sID_next")))
        .then(None)
        .when(pl.col("Adate_next").ge(pl.col("Ddate")))
        .then(None)
    )

    # Update Adate and Ddate based on sequential overlaps
    # and generate auxiliary info for additional intervals
    # We will explicitly use LazyFrame until the return to allow for optimisation
    updated_frame = (
        df.lazy()
        # ensure sorted
        .sort("sID", "Adate", "Ddate")
        # map up next row
        .with_columns(
            pl.col("sID", "fID", "Adate", "Ddate")
            .shift(-1)
            .map_alias(lambda x: f"{x}_next"),
        )
        # logic for new intervals
        # observation: If we have an overlap between successive intervals, we can update by shifting
        #   only the Ddate of the left interval and Adate of the right interval.
        #   In cases where the left interval completely covers the right interval, we will also need an extra
        #   interval, constructed between the two Ddates
        .with_columns(
            # updating the right hand time for the current row
            (
                reject_no_overlaps.when(pl.col("Adate_next").eq(pl.col("Adate")))
                .then(pl.col("Ddate"))
                .when(pl.col("Adate_next").gt(pl.col("Adate")))
                .then(pl.col("Adate_next"))
            ).alias("Ddate_new"),
            # updating the left hand time for the next entry
            (
                reject_no_overlaps.when(pl.col("Adate_next").eq(pl.col("Adate")))
                .then(pl.col("Ddate"))
                .when(pl.col("Adate_next").gt(pl.col("Adate")))
                .then(pl.col("Adate_next"))
            ).alias("Adate_next_new"),
            # a new row for encapsualted intervals
            (
                reject_no_overlaps.when(pl.col("Ddate") > pl.col("Ddate_next")).then(
                    pl.col("Ddate_next")
                )
            ).alias("ExtraAdate"),
            (
                reject_no_overlaps.when(pl.col("Ddate") > pl.col("Ddate_next")).then(
                    pl.col("Ddate")
                )
            ).alias("ExtraDdate"),
        )
        # prune columns
        .drop("sID_next", "fID_next", "Adate_next", "Ddate_next")
        # shift back Adate_next
        .with_columns(pl.col("Adate_next_new").shift(1))
        # Update the Adate and Ddate columns appropriately
        # If the corr. _new col is not null, use it, other wise don't
        .with_columns(
            (
                pl.when(pl.col("Ddate_new").is_not_null())
                .then(pl.col("Ddate_new"))
                .otherwise(pl.col("Ddate"))
            ).alias("Ddate"),
            (
                pl.when(pl.col("Adate_next_new").is_not_null())
                .then(pl.col("Adate_next_new"))
                .otherwise(pl.col("Adate"))
            ).alias("Adate"),
        )
        # Fix invalid intervals created by same-Adate situations with future overlaps
        .with_columns((pl.col("Ddate") < pl.col("Adate")).alias("Anomaly"))
        .with_columns(
            pl.when(pl.col("Anomaly"))
            .then(pl.col("Ddate"))
            .otherwise(pl.col("Adate"))
            .alias("Adate"),
            pl.when(pl.col("Anomaly").shift(-1))
            .then(pl.col("Ddate").shift(-1))
            .otherwise(pl.col("Ddate"))
            .alias("Ddate"),
        )
        .drop("Anomaly")
    )

    # join on extra intervals (formed by ExtraAdate and ExtraDdate)
    original_frame = updated_frame.select("sID", "fID", "Adate", "Ddate")
    extra_frame = updated_frame.select(
        pl.col("sID", "fID"),
        pl.col("ExtraAdate").alias("Adate"),
        pl.col("ExtraDdate").alias("Ddate"),
    ).filter(pl.all_horizontal(pl.col("Adate", "Ddate").is_not_null()))

    fixed_frame = pl.concat([original_frame, extra_frame]).sort("sID", "Adate", "Ddate")

    return fixed_frame.collect()


def fix_overlaps(df: pl.DataFrame, iters: int = 1, verbose=1):
    _t = 0
    if verbose:
        _t = tic()

    zfs = []
    for _i in range(iters):
        overlaps = scan_overlaps(df).collect()
        n_ov = overlaps.height
        known_dirty = overlaps.select("sID").to_series()
        zfs.append(df.filter(pl.col("sID").is_in(known_dirty).is_not()))
        df = df.filter(pl.col("sID").is_in(known_dirty))
        df = fix_overlaps_single_iter(df)
        if verbose:
            _t0, _t = _t, tic()
            print(
                "Iteration",
                _i,
                ":",
                df.height,
                "entries;",
                n_ov,
                "overlaps;",
                _t - _t0,
                "s",
            )
        if n_ov == 0:
            break
    print([z.height for z in zfs])
    return pl.concat([*zfs, df])
