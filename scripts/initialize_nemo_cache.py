#!/usr/bin/env python3
"""Clear and re-initialize the NeMo grammar cache.

Run this script after updating the text normalization pipeline to force
recompilation of the NeMo grammars. Compiled grammars are cached on disk
so subsequent startups are faster.
"""

import os
import shutil
import sys
from pathlib import Path

# Allow project imports regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wiki_tts.config import NEMO_GRAMMAR_CACHE
from wiki_tts.text import init_nemo

if __name__ == "__main__":
    # Clear the cache directory
    if os.path.exists(NEMO_GRAMMAR_CACHE):
        print(f"Removing {NEMO_GRAMMAR_CACHE} ...")
        shutil.rmtree(NEMO_GRAMMAR_CACHE)
    else:
        print(f"{NEMO_GRAMMAR_CACHE} does not exist, skipping removal.")

    # Trigger recompilation
    print("Initialising NeMo text processing (this may take >60 s) ...")
    init_nemo()
    print("Done.")
