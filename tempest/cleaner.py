import polars as pl
import datetime

from . import overlap_fixer as ovlfxr

from typing import Sequence, Hashable
from os import PathLike

_nulls = [
    "",
    "NA",
    "na",
    "Na",
    "N/A",
    "n/a",
    "N/a",
    "NaN",
    "''",
    " ",
    "NULL",
]


class DataHandlingError(Exception):
    pass


def ingest_csv(
    csv_path: PathLike | str,
    convert_dates: bool = False,
) -> pl.DataFrames:
    return pl.read_csv(
        csv_path, has_header=True, try_parse_dates=convert_dates, null_values=_nulls
    )


def clean_database(
    database: pl.DataFrame,
    delete_missing: bool = False,
    delete_errors: bool = False,
    convert_dates: bool = False,
    date_format: str = r"%Y-%m-%d",
    subject_id: str = "sID",
    facility_id: str = "fID",
    admission_date: str = "Adate",
    discharge_date="Ddate",
    subject_dtype: pl.DataType = pl.Utf8,
    facility_dtype: pl.DataType = pl.Utf8,
    retain_auxiliary_data: bool = True,
    n_iters: int = 100,
    verbose: bool = True,
) -> pl.DataFrame:
    report = standardise_column_names(
        database, subject_id, facility_id, admission_date, discharge_date, verbose
    )

    report = coerce_data_types(
        report, convert_dates, date_format, subject_dtype, facility_dtype, verbose
    )

    # Trim auxiliary data
    if not retain_auxiliary_data:
        if verbose:
            print("Trimming auxiliary data...")
        report = report.select(pl.col("sID", "fID", "Adate", "Ddate"))

    # Check and clean missing values
    report = clean_missing_values(
        report, delete_missing=delete_missing, verbose=verbose
    )

    # Check erroneous records
    report = clean_erroneous_records(
        report, delete_errors=delete_errors, verbose=verbose
    )

    # remove row duplicates
    if verbose:
        print("Removing duplicate records...")
    report = report.unique()

    # Fix overlapping stays
    report = fix_all_overlaps(report, n_iters, verbose)

    return report


def standardise_column_names(
    df: pl.DataFrame,
    subject_id: str = "sID",
    facility_id: str = "fID",
    admission_date: str = "Adate",
    discharge_date: str = "Ddate",
    verbose: bool = True,
) -> pl.DataFrame:
    """Check and standardise column names for further processing"""

    # Check column existence
    if verbose:
        print("Checking existence of columns...")
    expected_cols = {subject_id, facility_id, admission_date, discharge_date}
    found_cols = set(df.columns)
    missing_cols = expected_cols.difference(found_cols)
    if len(missing_cols):
        raise DataHandlingError(
            f"Column(s) {', '.join(missing_cols)} provided as argument were not found in the database."
        )
    elif verbose:
        print("Column existence OK.")

    # Standardise column names
    if verbose:
        print("Standardising column names...")
    return df.rename(
        {
            subject_id: "sID",
            facility_id: "fID",
            admission_date: "Adate",
            discharge_date: "Ddate",
        }
    )


def coerce_data_types(
    database: pl.DataFrame,
    convert_dates: bool = False,
    date_format: str = r"%Y-%m-%d",
    subject_dtype: pl.DataType = pl.Utf8,
    facility_dtype: pl.DataType = pl.Utf8,
    verbose: bool = True,
) -> pl.DataFrame:
    # Check data format, column names, variable format, parse dates
    if verbose:
        print("Coercing types...")
    if convert_dates:
        if verbose:
            print("Converting dates...")
        date_expressions = [
            pl.col("Adate").str.strptime(pl.Datetime, format=date_format),
            pl.col("Ddate").str.strptime(pl.Datetime, format=date_format),
        ]
    else:
        # do nothing
        date_expressions = [pl.col("Adate"), pl.col("Ddate")]
    # Coerce types
    database = database.with_columns(
        [
            pl.col("sID").cast(subject_dtype),
            pl.col("fID").cast(facility_dtype),
            *date_expressions,
        ]
    )
    if verbose:
        print("Type coercion done.")
    return database


def clean_missing_values(
    database: pl.DataFrame, delete_missing: bool = False, verbose: bool = True
) -> pl.DataFrame:
    """Checks for and potentially deletes recods with missing values"""
    # Check for missing values
    if verbose:
        print("Checking for missing values...")
    missing_records = database.filter(
        pl.any(pl.col("*").is_null()) | pl.any(pl.col("sID", "fID").str.strip() == "")
    )
    if len(missing_records):
        if verbose:
            print(f"Found {len(missing_records)} records with missing values.")
        if not delete_missing:
            raise DataHandlingError(
                "Please deal with these missing values or set argument delete_missing to 'record' or 'subject'."
            )
        elif delete_missing == "record":
            if verbose:
                print("Deleting missing records...")
            return database.filter(pl.all(pl.col("*").is_not_null()))
        elif delete_missing == "subject":
            if verbose:
                print("Deleting records of subjects with any missing records...")
            subjects = missing_records.select(pl.col("sID")).to_series()
            return database.filter(~pl.col("subject").is_in(subjects))
        else:
            raise DataHandlingError(
                f"Unknown delete_missing value: {delete_missing}. Acceptable values: 'record', 'subject'."
            )

    # no missing return as-is
    return database


def clean_erroneous_records(
    database: pl.DataFrame, delete_errors: bool = False, verbose: bool = True
) -> pl.DataFrame:
    """Checks for and potnetially deletes records which are erroneous

    Erroneous records are when the discharge date is recrded as before the admission date
    """
    if verbose:
        print("Checking for erroneous records...")
    erroneous_records = database.filter(pl.col("Adate") > pl.col("Ddate"))
    if len(erroneous_records):
        if verbose:
            print(f"Found {len(erroneous_records)} records with date errors.")
        if not delete_errors:
            raise DataHandlingError(
                "Please deal with these errors or set argument delete_errors to 'record' or 'subject'."
            )
        elif delete_errors == "record":
            if verbose:
                print("Deleting records with date errors...")
            return database.filter((pl.col("Adate") > pl.col("Ddate")).is_not())
        elif delete_errors == "subject":
            if verbose:
                print("Deleting records of subjects with date errors...")
            subjects = erroneous_records.select(pl.col("sID")).to_series()
            return database.filter(~pl.col("subject").is_in(subjects))
    # no errors, return as-is
    return database


def fix_all_overlaps(
    database: pl.DataFrame, n_iters: int = 100, verbose: bool = True
) -> pl.DataFrame:
    if verbose:
        print("Finding and fixing overlapping records...")

    database = ovlfxr.fix_overlaps(database, iters=n_iters, verbose=verbose)

    if verbose:
        n_overlaps = ovlfxr.num_overlaps(database)
        print(n_overlaps, "overlaps remaining after iterations...")

    return database


_REF_DATE = datetime.datetime(year=2017, month=3, day=1)


def normalise_dates(
    database: pl.DataFrame,
    cols: Sequence[Hashable],
    ref_date: datetime.datetime = _REF_DATE,
) -> pl.DataFrame:
    """Normalises Datetime columns to the number of days past a given reference date"""
    return database.with_columns(
        *((pl.col(col) - ref_date).dt.seconds() / 60 / 60 / 24 for col in cols)
    )
