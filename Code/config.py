"""
config.py
Shared constants and configuration for the exoplanet detection pipeline.
"""

# --- Data acquisition ---
DEFAULT_MISSION = "TESS"
DEFAULT_AUTHOR = "SPOC"          # Science Processing Operations Center pipeline products
DEFAULT_FLUX_COLUMN = "pdcsap_flux"  # Pre-search Data Conditioning SAP flux (systematics-corrected)

# --- Preprocessing ---
SIGMA_CLIP_LOWER = 8
SIGMA_CLIP_UPPER = 5
FLATTEN_WINDOW_LENGTH = 401       # odd integer, cadences, ~ a few transit durations wide
FLATTEN_POLYORDER = 2
NAN_POLICY = "drop"

# --- BLS / period search ---
MIN_PERIOD_DAYS = 1.0
MAX_PERIOD_DAYS = 20.0
N_PERIODS = 15000
MIN_DURATION_HOURS = 0.5
MAX_DURATION_HOURS = 10.0
N_DURATIONS = 20
SNR_DETECTION_THRESHOLD = 7.0     # candidates below this are discarded as noise

# --- Classification categories ---
CLASSES = ["transit", "eclipsing_binary", "blend", "other"]

# --- Random seed for reproducibility ---
RANDOM_STATE = 42
