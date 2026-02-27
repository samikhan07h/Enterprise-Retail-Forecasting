import pandas as pd
from sqlalchemy import create_engine

def load_from_sql(connection_string, table_name):
    engine = create_engine(connection_string)
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, engine)