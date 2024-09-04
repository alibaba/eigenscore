import getpass
import os
import sys

__USERNAME = getpass.getuser()
_BASE_DIR = f'/disk1/chenchao/Code/hallucination_detection/data'
MODEL_PATH = f'{_BASE_DIR}/weights/'
DATA_FOLDER = os.path.join(_BASE_DIR, 'datasets')
GENERATION_FOLDER = os.path.join(_BASE_DIR, 'output')
os.makedirs(GENERATION_FOLDER, exist_ok=True)

