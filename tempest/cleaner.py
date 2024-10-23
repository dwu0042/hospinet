import polars as pl
from . import overlap_fixer as ovlfxr

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
    csv_path,
    convert_dates=False,
):
    return pl.read_csv(
        csv_path, has_header=True, try_parse_dates=convert_dates, null_values=_nulls
    )


def clean_database(
    database: pl.DataFrame,
    delete_missing=False,
    delete_errors=False,
    convert_dates=False,
    date_format=r"%Y-%m-%d",
    subject_id="sID",
    facility_id="fID",
    admission_date="Adate",
    discharge_date="Ddate",
    subject_dtype=pl.Utf8,
    facility_dtype=pl.Utf8,
    retain_auxiliary_data=True,
    n_iters=100,
    verbose=True,
):
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
    subject_id="sID",
    facility_id="fID",
    admission_date="Adate",
    discharge_date="Ddate",
    verbose=True,
):
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
    convert_dates=False,
    date_format=r"%Y-%m-%d",
    subject_dtype=pl.Utf8,
    facility_dtype=pl.Utf8,
    verbose=True,
):
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


def clean_missing_values(database: pl.DataFrame, delete_missing=False, verbose=True):
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


def clean_erroneous_records(database: pl.DataFrame, delete_errors=False, verbose=True):
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


def fix_all_overlaps(database: pl.DataFrame, n_iters=100, verbose=True):
    if verbose:
        print("Finding and fixing overlapping records...")

    database = ovlfxr.fix_overlaps(database, iters=n_iters, verbose=verbose)

    if verbose:
        n_overlaps = ovlfxr.num_overlaps(database)
        print(n_overlaps, "overlaps remaining after iterations...")

    return database
