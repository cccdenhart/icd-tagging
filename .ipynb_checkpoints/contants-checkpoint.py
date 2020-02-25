from dotenv import load_dotenv
from pyathena import connect
import os

# define project location
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

# load local environment variables
ENV_PATH = os.path.join(PROJ_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

cursor = connect(aws_access_key_id=os.getenv("ACCESS_KEY"),
                 aws_secret_access_key=os.getenv("SECRET_KEY"),
                 s3_staging_dir=os.getenv("S3_DIR"),
                 region_name=os.getenv("REGION_NAME"))
