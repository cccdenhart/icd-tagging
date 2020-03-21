from dotenv import load_dotenv
import os

# define project location
PROJ_DIR: str = os.path.dirname(os.path.abspath(__file__))

# load local environment variables
env_name: str = ".env"
load_dotenv(dotenv_path=os.path.join(PROJ_DIR, env_name))

# preprocessing constants
ICD_COLNAME: str = "icd9_codes"
NOTE_COLNAME: str = "note"
LABEL_FN: str = "labels.csv"
TFIDF_FN: str = "tfidf.csv"
