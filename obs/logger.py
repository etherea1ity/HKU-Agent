import json, os, uuid, datetime
from typing import Any, Dict, Optional

def new_run_id()-> str:
    return uuid.uuid4().hex

class JsonLogger:
    def __init__(self, log_path: str, run_id: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.run_id = run_id

    def emit(self, event: str, stage: str = "", attrs: Optional[Dict[str, Any]] = None):
        rec = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "event": event,
            "stage": stage,
            "attrs": attrs or {},
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

