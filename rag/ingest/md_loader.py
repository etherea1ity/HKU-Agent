from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class MdDoc:
    source_path: str
    title: str
    text: str
    meta: Dict[str, Any]

def load_md(path: str) -> MdDoc:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    title = os.path.basename(path)
    return MdDoc(source_path=path, title=title, text=txt, meta={})