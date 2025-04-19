import logging
import sys
import os

# Ensure the logs directory exists
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Define log file path
log_file_path = os.path.join(LOG_DIR, "run_log.txt")

# Setup logger
log = logging.getLogger("short4me")
log.setLevel(logging.DEBUG)

# File handler for logs/run_log.txt
file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers (prevent duplication)
if not log.handlers:
    log.addHandler(file_handler)
    log.addHandler(console_handler)
