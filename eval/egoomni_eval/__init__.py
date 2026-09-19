"""egoOmni evaluation harness. Pure-python core (json/subprocess only) so it imports in every model env."""
import os
ROOT = os.environ.get("EGO_ROOT", "/ai4good1-shared/liyu/egoOmni_baselines")
EVAL_DIR = os.path.join(ROOT, "eval")
DATA_DIR = os.environ.get("EGO_DATA", "/ai4good1-shared/liyu/egoOmni")
QA_PATH = os.path.join(DATA_DIR, "test", "qa.json")
CLIPS_META = os.path.join(EVAL_DIR, "clips_meta.json")
REPOS = os.path.join(ROOT, "repos")
