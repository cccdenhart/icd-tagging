"""Loads constants, allowing for easy accessibility in other files."""
import os
from typing import Callable, Optional

from dotenv import load_dotenv
from pyathena import connect
from pyathena.connection import Connection

# get the project root directory
PROJ_DIR: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

# load environment variables
env_path: str = os.path.join(PROJ_DIR, ".env")
load_dotenv(dotenv_path=env_path)
ENV: Callable[[str], Optional[str]] = os.getenv

# establish pyathena connection
CONN: Connection = connect(aws_access_key_id=ENV("ACCESS_KEY"),
                           aws_secret_access_key=ENV("SECRET_KEY"),
                           s3_staging_dir=ENV("S3_DIR"),
                           region_name=ENV("REGION_NAME"))
