import os
import sys

# Ensure repository root and Articles package are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTICLES_DIR = os.path.join(PROJECT_ROOT, "Articles")

for path in (PROJECT_ROOT, ARTICLES_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
