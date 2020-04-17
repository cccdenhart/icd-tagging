import os

import pandas as pd
import pyathena
from pyathena.connection import Connection as AthenaConn
from pyathena.util import as_pandas
from dotenv import load_dotenv

from icd9.icd9 import ICD9

""" --- Utility constants --- """
# define project location
PROJ_DIR: str = os.path.dirname(os.path.abspath(__file__))

# load the icd9 tree
TREE = ICD9(os.path.join(PROJ_DIR, "icd9", "codes.json"))

# name for range of V codes
V_CODE = "V01-V91"


""" --- Utility functions --- """
def get_conn() -> AthenaConn:
    """Allow passing of pyathena connection."""
    # load the environment
    env_name = ".env"
    load_dotenv(dotenv_path=os.path.join(PROJ_DIR, env_name))
    return pyathena.connect(aws_access_key_id=os.getenv("ACCESS_KEY"),
                            aws_secret_access_key=os.getenv("SECRET_KEY"),
                            s3_staging_dir=os.getenv("S3_DIR"),
                            region_name=os.getenv("REGION_NAME"))


def query_aws(query: str) -> pd.DataFrame:
    """Execute a query on the Athena database and return as pandas."""
    with get_conn() as conn:
        cursor = conn.cursor()
        df = as_pandas(cursor.execute(query))
    return df


