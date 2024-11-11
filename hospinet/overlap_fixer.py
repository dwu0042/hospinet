"""This submodule provides utility functions to the `cleaner` submodule to correct overlapping patient admissions"""

import logging
import re
import polars as pl

from time import perf_counter as tic

logger = logging.getLogger("hospinet")


def scan_overlaps(df: pl.DataFrame) -> pl.LazyFrame:
    """Constructs a query that can resolve to the overlaps present in the given dataframe

    Overlaps are records where, for the same patient, the admission date for a record is before the discharge date of the previous record.

    Args:
        df (pl.DataFrame): Database to scan for overlaps

    Returns:
        pl.LazyFrame: Query that can be .collect() -ed to return a dataframe of overlaps
    """
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


def num_overlaps(df: pl.DataFrame) -> int:
    """Get the number of overlaps in a dataframe

    Overlaps are records where, for the same patient, the admission date for a record is before the discharge date of the previous record.

    Args:
        df (pl.DataFrame): Database to scan for overlaps

    Returns:
        int: number of overlapping records

    """
    return scan_overlaps(df).collect().height


def fix_overlaps_single_iter(df: pl.DataFrame) -> pl.DataFrame:
    """Performs one iteration of overlap correction on successive presence events

    For two successive admissions of a patient (as sorted by Adate) we have 4 cases:

    1. The left and right intervals do not overlap (left Ddate < right Adate)
        - no fix is needed
    2. The left Ddate is after the Adate of the right interval, but before the right Ddate
        - we move the Ddate of the left interval back
    3. The left Ddate is after the right Ddate (one interval completely encapsulates the other)
        - we move the Ddate of the left interval back
        - we create a new interval from the right Ddate to the left Ddate

    This single iteration does not necessarily fix all overlaps.

    Args:
        df (pl.DataFrame): Dataframe to fix overlaps in

    Returns:
        pl.DataFrame: Dataframe with one iteration of overlap fixes

    """

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
            .name.map(lambda x: f"{x}_next"),
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
            # a new row for encapsulated intervals
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
        # If the corr. _new col is not null, use it, otherwise don't
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


def fix_overlaps(df: pl.DataFrame, iters: int = 1) -> pl.DataFrame:
    """Fixes overlapping patient records in the given database


    Also see `fix_overlaps_single_iter` for the overlap correction logic

    Args:
        df (pl.DataFrame): Database with overlaps to fix
        iters (int, optional): Number of iterations to run. Defaults to 1.

    Returns:
        pl.DataFrame: Database with overlaps fixed
    """

    timer_start = 0
    timer_start = tic()

    clean_records = []
    logger.info(f"Attempting up to {iters} iterations")
    for iteration in range(iters):
        overlaps = scan_overlaps(df).collect()
        n_overlaps = overlaps.height
        patients_with_overlaps = overlaps.select("sID").to_series()
        clean_records.append(
            df.filter(pl.col("sID").is_in(patients_with_overlaps).not_())
        )
        df = df.filter(pl.col("sID").is_in(patients_with_overlaps))
        df = fix_overlaps_single_iter(df)
        timer_old, timer_start = timer_start, tic()
        logger.info(
            f"Iteration {iteration}: {df.height} entries; {n_overlaps} overlaps; {timer_start - timer_old} s"
        )
        if n_overlaps == 0:
            break

    # print number of non overlapping records; group zeros into "...[count]"
    non_overlaps = str([z.height for z in clean_records])
    non_overlaps_grouped_zeros = re.sub(
        r"( 0,){2,}", lambda m: rf" 0...{{{m.group(0).count('0')}x}},", non_overlaps
    )
    logger.info("History of non-overlapping patient records:")
    logger.info(non_overlaps_grouped_zeros)

    # join the clean and corrected records
    return pl.concat([*clean_records, df])
