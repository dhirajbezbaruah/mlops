"""Top level package for Feddit Analyzer."""

import os
from importlib import metadata
from pathlib import Path

__version__ = metadata.version("feddit_analyzer")

WORKDIR = Path(os.getenv("WORKDIR", Path.cwd()))
BASEPATH = Path(__file__).parent
