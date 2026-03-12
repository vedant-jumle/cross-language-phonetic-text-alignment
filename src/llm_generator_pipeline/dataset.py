from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

SPLITS = {"train", "val", "test"}

class NameDataset(Dataset):
    def __init__(self, path: str, split: str):
        if split not in SPLITS:
            raise ValueError(f"Invalid split: {split}!")
        
        self.path = path
        
        if not self.path.exists():
            raise FileNotFoundError(f"Can't find path: {path}")
        
        self.split = split
        self.records: list[dict[str, Any]] = []
        
        with self.path.open("r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                if not line: 
                    continue
                
                try: 
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON line at {line_num} of {self.path}: {e}!") from e
                    
                if record.get("split") != split:
                    continue
                
                # don't forget to add some validation function for records
                self.validate_record(record, line_num)
                self.records.append(record)
        
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]
    
    