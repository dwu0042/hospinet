import polars as pl
# import connectorx as cx

URI = "mssql://localhost/DATABASE?trusted_connection=true"

skim_early_query = """
SELECT
    sID,
    fID,
    Adate,
    Ddate
FROM
    TableName
ORDER BY Adate ASC
    OFFSET 0 ROWS
    FETCH NEXT 1000 ROWS ONLY;
"""

df = pl.read_database(skim_early_query, URI)

