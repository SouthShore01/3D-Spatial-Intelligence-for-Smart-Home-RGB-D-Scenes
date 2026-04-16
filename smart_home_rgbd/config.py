from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DEFAULT_SUNRGBD_ROOT = PROJECT_ROOT / "SUNRGBD"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_INDEX_CSV = DEFAULT_ARTIFACTS_DIR / "sunrgbd_scene_index.csv"
DEFAULT_STATS_JSON = DEFAULT_ARTIFACTS_DIR / "sunrgbd_scene_stats.json"
DEFAULT_MANUAL_LABELS_CSV = DEFAULT_ARTIFACTS_DIR / "manual_state_labels_seed.csv"
DEFAULT_INSTANCE_MANIFEST_JSON = DEFAULT_ARTIFACTS_DIR / "manual_state_instance_manifest.json"
DEFAULT_INSTANCE_LABELS_JSON = DEFAULT_ARTIFACTS_DIR / "manual_object_state_labels.json"
DEFAULT_LABEL_UI_PORT = 8765
