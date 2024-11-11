"""This submodule provides utilities for validating and correcting a database of patient admissions."""

import datetime
import logging

import polars as pl

from . import overlap_fixer as ovlfxr

from typing import Sequence, Hashable
from os import PathLike

logger = logging.getLogger("hospinet")

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
        pl.DataFrame: Dataframe representing the ingested csv
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
        delete_missing (str | bool, optional): One of "record", "subject" or False. If "record", delete records which have missing values; if "subject", deletes records belonging to subjects which have at least one record that has a missing value; if False raises an exception if any records have missing values. Defaults to False.
        delete_errors (str | bool, optional): One of "record", "subject" or False. If "record", delete erroneous records; if "subject", deletes records belonging to subjects which have at least one erroneous record; if False raises an exception if erroneous records exist. Defaults to False.
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

    if verbose:
        logger.setLevel(logging.INFO)

    database = standardise_column_names(
        database=database,
        subject_id=subject_id,
        facility_id=facility_id,
        admission_date=admission_date,
        discharge_date=discharge_date,
    )

    database = coerce_data_types(
        database=database,
        manually_convert_dates=manually_convert_dates,
        date_format=date_format,
        subject_dtype=subject_dtype,
        facility_dtype=facility_dtype,
    )

    # Trim auxiliary data
    if not retain_auxiliary_data:
        logger.info("Trimming auxiliary data...")
        database = database.select(pl.col("sID", "fID", "Adate", "Ddate"))

    # Check and clean missing values
    database = clean_missing_values(
        database=database,
        delete_missing=delete_missing,
    )

    # Check erroneous records
    database = clean_erroneous_records(
        database=database,
        delete_errors=delete_errors,
    )

    # remove row duplicates
    logger.info("Removing duplicate records...")
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
) -> pl.DataFrame:
    """Check and standardise column names for further processing

    Args:
        database (pl.DataFrame): Database (polars dataframe) of patient admissions. Columns have at least: patient, facility, admission time, discharge time.
        subject_id (str, optional): Column name in the database that corresponds to the patient (subject). Defaults to "sID".
        facility_id (str, optional): Column name in the database that corresponds to the hospital (facility). Defaults to "fID".
        admission_date (str, optional): Column name in the database that corresponds to admission date/time. Defaults to "Adate".
        discharge_date (str, optional): Column name in the database that corresponds to discharge date/time. Defaults to "Ddate".

    Raises:
        DataHandlingError: If there is a missing column from the given columns.

    Returns:
        pl.DataFrame: Database with normalised column names
    """
    # Check column existence
    logger.info("Checking existence of columns...")
    expected_cols = {subject_id, facility_id, admission_date, discharge_date}
    found_cols = set(database.columns)
    missing_cols = expected_cols.difference(found_cols)
    if len(missing_cols):
        error_message = f"Column(s) {', '.join(missing_cols)} provided as argument were not found in the database."
        logger.error(error_message)
        raise DataHandlingError(error_message)
    else:
        logger.info("Column existence OK.")

    # Standardise column names
    logger.info("Standardising column names...")
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
) -> pl.DataFrame:
    """Cast data types of the core columns to standard (given) types

    Args:
        database (pl.DataFrame): Database (polars dataframe) of patient admissions. Columns have at least: patient, facility, admission time, discharge time.
        manually_convert_dates (bool, optional): if True, converts admission and discharge date columns from string type to datetime type manually, must be provided with a date_format; if False, does not modify those columns. Defaults to False.
        date_format (str, optional): date format to expect if manually_convert_dates is True. Defaults to r"%Y-%m-%d".
        subject_dtype (pl.DataType, optional): Polars datatype to coerce patient IDs to. Defaults to pl.Utf8.
        facility_dtype (pl.DataType, optional): Polars datatype to coerce hospital IDs to. Defaults to pl.Utf8.

    Returns:
        pl.DataFrame: Database with normalised column datatypes
    """
    # Check data format, column names, variable format, parse dates
    logger.info("Coercing types...")
    if manually_convert_dates:
        logger.info(f"Manually converting dates from format {date_format}...")
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
    logger.info("Type coercion done.")
    return database


def clean_missing_values(
    database: pl.DataFrame,
    delete_missing: str | bool = False,
) -> pl.DataFrame:
    """Checks for and potentially deletes records with missing values

    Args:
        database (pl.DataFrame): Database (polars dataframe) of patient admissions. Columns have at least: patient, facility, admission time, discharge time.
        delete_missing (str | bool, optional): One of "record", "subject" or False. If "record", delete records which have missing values; if "subject", deletes records belonging to subjects which have at least one record that has a missing value; if False raises an exception if any records have missing values. Defaults to False.

    Raises:
        DataHandlingError: if delete_missing was set to False and missing records were found.

    Returns:
        pl.DataFrame: Database with missing values fixed
    """
    # Check for missing values
    logger.info("Checking for missing values...")
    missing_records = database.filter(
        (
            pl.any_horizontal(pl.all().is_null())
            | pl.any_horizontal(pl.col("sID", "fID").str.strip_chars() == "")
        )
    )
    if len(missing_records):
        logger.info(f"Found {len(missing_records)} records with missing values.")
        match delete_missing:
            case False:
                raise DataHandlingError(
                    "Please deal with these missing values or set argument delete_missing to 'record' or 'subject'."
                )
            case "record":
                logger.info("Deleting missing records...")
                return database.filter(pl.all(pl.col("*").is_not_null()))
            case "subject":
                logger.info("Deleting records of subjects with any missing records...")
                subjects = missing_records.select(pl.col("sID")).to_series()
                return database.filter(~pl.col("subject").is_in(subjects))
            case _:
                raise ValueError(
                    f"""Unknown delete_missing value: {delete_missing}. Acceptable values: "record", "subject", False."""
                )

    # no missing return as-is
    return database


def clean_erroneous_records(
    database: pl.DataFrame,
    delete_errors: str | bool = False,
) -> pl.DataFrame:
    """Checks for and potentially deletes records which are erroneous

    Erroneous records are when the discharge date is recorded as before the admission date

    Args:
        database (pl.DataFrame): Database (polars dataframe) of patient admissions. Columns have at least: patient, facility, admission time, discharge time.
        delete_errors (str | bool, optional): One of "record", "subject" or False. If "record", delete erroneous records; if "subject", deletes records belonging to subjects which have at least one erroneous record; if False raises an exception if erroneous records exist. Defaults to False.

    Raises:
        DataHandlingError: if delete_errors is False, and any erroneous records are found.

    Returns:
        pl.DataFrame: Database with erroneous records fixed
    """
    logger.info("Checking for erroneous records...")
    erroneous_records = database.filter(pl.col("Adate") > pl.col("Ddate"))
    if len(erroneous_records):
        logger.info(f"Found {len(erroneous_records)} records with date errors.")
        match delete_errors:
            case False:
                raise DataHandlingError(
                    "Please deal with these errors or set argument delete_errors to 'record' or 'subject'."
                )
            case "record":
                logger.info("Deleting records with date errors...")
                return database.filter((pl.col("Adate") > pl.col("Ddate")).is_not())
            case "subject":
                logger.info("Deleting records of subjects with date errors...")
                subjects = erroneous_records.select(pl.col("sID")).to_series()
                return database.filter(~pl.col("subject").is_in(subjects))
            case _:
                raise ValueError(
                    f"""Unknown delete_errors value: {delete_errors}. Acceptable values: "record", "subject", False."""
                )

    # no errors, return as-is
    return database


def fix_all_overlaps(
    database: pl.DataFrame, n_iters: int = 100, log_iteration_status: bool = True
) -> pl.DataFrame:
    """Fixes overlapping records in the database.

    Records for a given patient overlap if the admission date for one record is before the discharge date for another.

    See also `overlap_fixer.fix_overlaps`.

    Args:
        database (pl.DataFrame): Database (polars dataframe) of patient admissions. Columns have at least: patient, facility, admission time, discharge time.
        n_iters (int, optional): Maximum number of iterations of overlap fixing. Defaults to 100.
        log_iteration_status (bool, optional): if True, logs the number of overlaps at the end of each iteration

    Returns:
        pl.DataFrame: Database with overlaps corrected
    """
    logger.info("Finding and fixing overlapping records...")

    database = ovlfxr.fix_overlaps(
        database,
        iters=n_iters,
    )

    if log_iteration_status:
        n_overlaps = ovlfxr.num_overlaps(database)
        logger.info(f"{n_overlaps} overlaps remaining after iterations")

    return database


_REF_DATE = datetime.datetime(year=2017, month=3, day=1)


def normalise_dates(
    database: pl.DataFrame,
    cols: Sequence[Hashable],
    ref_date: datetime.datetime = _REF_DATE,
) -> pl.DataFrame:
    """Normalises given Datetime columns to the number of days past a given reference date

    Args:
        database (pl.DataFrame): Database (polars dataframe) to be normalised
        cols (Sequence[Hashable]): Column names of the datetime columns to convert from datetime to numeric (number of days past ref date)
        ref_date (datetime.datetime, optional): Reference date to normalise datetimes against. Defaults to 1 March 2017 00:00.

    Returns:
        pl.DataFrame: Database with the given columns normalised
    """
    """"""
    return database.with_columns(
        *((pl.col(col) - ref_date).dt.total_seconds() / 60 / 60 / 24 for col in cols)
    )
