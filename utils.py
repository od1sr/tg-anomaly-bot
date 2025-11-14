import os
import time
import uuid
from config import SAVE_DIR

os.makedirs(SAVE_DIR, exist_ok=True)

def _make_filename(ext):
    ts = int(time.time())
    uid = uuid.uuid4().hex[:12]
    if not ext:
        ext = ".jpg"
    return f"{ts}_{uid}{ext}"