import os
from icd9.icd9 import ICD9

# define project location
PROJ_DIR: str = os.path.dirname(os.path.abspath(__file__))

# load local environment variables
ENV_NAME: str = ".env"

# preprocessing constants
ICD_COLNAME: str = "icd9_codes"
NOTE_COLNAME: str = "note"
LABEL_FN: str = "labels.csv"
TFIDF_FN: str = "tfidf.csv"

# load the icd9 tree
TREE = ICD9(os.path.join(PROJ_DIR, "icd9", "codes.json"))
