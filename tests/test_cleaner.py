import polars as pl
from tempest import cleaner


def test_overall():
    database = cleaner.ingest_csv("fakedb_werr.data.table")
    cleaned_db = cleaner.clean_database(
        database,
        subject_id="subject",
        facility_id="facility",
        admission_date="admission",
        discharge_date="discharge",
        convert_dates=True,
        delete_missing="record",
        delete_errors="record",
    )

    assert cleaned_db.filter(pl.any(pl.col("*").is_null())).is_empty()
