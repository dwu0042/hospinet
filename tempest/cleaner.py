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
    """Expected Exception when handling data"""


def ingest_csv(
    csv_path: PathLike | str,
    convert_dates: bool = False,
) -> pl.DataFrame:
    """Reads a CSV, with null value interpretation and optional date parsing

    Args:
        csv_path (PathLike | str): path to the csv to read
        convert_dates (bool, optional): if True, polars automagically attempts to convert date-like columns. Defaults to False.

    Returns:
        pl.DataFrame: _description_
    """    
    return pl.read_csv(
        csv_path, has_header=True, try_parse_dates=convert_dates, null_values=_nulls
    )


def clean_database(
    database: pl.DataFrame,
    delete_missing: bool = False,
    delete_errors: bool = False,
    manually_convert_dates: bool = False,
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
    """Cleans a database of patient admissions
    
    Standardises column names, coerces columns to standard data types,
    removes missing and erroneous values, and fixes overlapping admissions.

    Args:
        database (pl.DataFrame): Database (polars dataframe) of patient admissions. Columns should be at least: patient, facility, admission time, discharge time.
        delete_missing (bool, optional): if True, deletes rows with missing values; otherwise raises an Exception if any rows with missing data are found. Defaults to False.
        delete_errors (bool, optional): if True, deletes rows with a discharge date before the admission date; otherwise raises an Exception is any such rows are found. Defaults to False.
        manually_convert_dates (bool, optional): if True, converts admission and discharge date columns from string type to datetime type manually, must be provided with a date_format; if False, does not modify those columns. Defaults to False.
        date_format (str, optional): date format to expect if manually_convert_dates is True. Defaults to r"%Y-%m-%d".
        subject_id (str, optional): Column name in the database that corresponds to the patient (subject). Defaults to "sID".
        facility_id (str, optional): Column name in the database that corresponds to the hospital (facility). Defaults to "fID".
        admission_date (str, optional): Column name in the database that corresponds to admission date/time. Defaults to "Adate".
        discharge_date (str, optional): Column name in the database that corresponds to discharge date/time. Defaults to "Ddate".
        subject_dtype (pl.DataType, optional): Polars datatype to coerce patient IDs to. Defaults to pl.Utf8.
        facility_dtype (pl.DataType, optional): Polars datatype to coerce hospital IDs to. Defaults to pl.Utf8.
        retain_auxiliary_data (bool, optional): if True, retains columns that are not subject, facility, admission and discharge dates; otherwise drops those columns. Defaults to True.
        n_iters (int, optional): Maximum number of iterations of overlap fixing. Defaults to 100.
        verbose (bool, optional): if True, prints informational messages to STDOUT, otherwise run silently. Defaults to True.

    Returns:
        pl.DataFrame: Cleaned database
    """    

    database = standardise_column_names(
        database=database, 
        subject_id=subject_id, 
        facility_id=facility_id, 
        admission_date=admission_date, 
        discharge_date=discharge_date, 
        verbose=verbose,
    )

    database = coerce_data_types(
        database=database, 
        manually_convert_dates=manually_convert_dates, 
        date_format=date_format, 
        subject_dtype=subject_dtype, 
        facility_dtype=facility_dtype, 
        verbose=verbose,
    )

    # Trim auxiliary data
    if not retain_auxiliary_data:
        if verbose:
            print("Trimming auxiliary data...")
        database = database.select(pl.col("sID", "fID", "Adate", "Ddate"))

    # Check and clean missing values
    database = clean_missing_values(
        database=database, 
        delete_missing=delete_missing, 
        verbose=verbose,
    )

    # Check erroneous records
    database = clean_erroneous_records(
        database=database, 
        delete_errors=delete_errors, 
        verbose=verbose,
    )

    # remove row duplicates
    if verbose:
        print("Removing duplicate records...")
    database = database.unique()

    # Fix overlapping stays
    database = fix_all_overlaps(database, n_iters, verbose)

    return database


def standardise_column_names(
    database: pl.DataFrame,
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
    found_cols = set(database.columns)
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
    return database.rename(
        {
            subject_id: "sID",
            facility_id: "fID",
            admission_date: "Adate",
            discharge_date: "Ddate",
        }
    )


def coerce_data_types(
    database: pl.DataFrame,
    manually_convert_dates: bool = False,
    date_format: str = r"%Y-%m-%d",
    subject_dtype: pl.DataType = pl.Utf8,
    facility_dtype: pl.DataType = pl.Utf8,
    verbose: bool = True,
) -> pl.DataFrame:
    # Check data format, column names, variable format, parse dates
    if verbose:
        print("Coercing types...")
    if manually_convert_dates:
        if verbose:
            print(f"Manually converting dates from format {date_format}...")
        date_expressions = [
            pl.col("Adate").str.strptime(pl.Datetime, format=date_format),
            pl.col("Ddate").str.strptime(pl.Datetime, format=date_format),
        ]
    else:
        # do nothing
        date_expressions = []
    # Coerce types
    database = database.with_columns(
        pl.col("sID").cast(subject_dtype),
        pl.col("fID").cast(facility_dtype),
        *date_expressions,
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
    missing_records = database.filter((
        pl.any_horizontal(pl.all().is_null()) 
        | pl.any_horizontal(pl.col("sID", "fID").str.strip_chars() == "")
    ))
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
    """Checks for and potentially deletes records which are erroneous

    Erroneous records are when the discharge date is recorded as before the admission date
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
        *((pl.col(col) - ref_date).dt.total_seconds() / 60 / 60 / 24 for col in cols)
    )
