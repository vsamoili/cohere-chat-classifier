from pathlib import Path

from dotenv import load_dotenv
from yaml import load, Loader

# Load config from yaml
with open(Path(__file__).parent.parent / 'config.yaml', 'r') as fp:
    config = load(fp, Loader=Loader)


# Load env vars from .env file
load_dotenv()
