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
    
    @staticmethod
    def validate_record(record: dict[str, Any], line_num: int) -> None:
    
        required_top = {"entity_id", "split", "anchor", "positives"}
        missing_top = required_top - set(record.keys())
        
        if missing_top:
            raise ValueError(f"Missing top-level keys {missing_top} on line {line_num}.")
        
        if not isinstance(record["entity_id"], str):
            raise ValueError(f"'entity_id' must be a str on line {line_num}.")
        if record["split"] not in SPLITS:
            raise ValueError(f"Invalid split {record["split"]} on line {line_num}.")
        
        anchor = record["anchor"]
        if not isinstance(anchor, dict):
            raise ValueError(f"'anchor' must be a dict on line {line_num}.")    
        if "text" not in anchor or "bytes" not in anchor:
            raise ValueError(f"'anchor' missing keys on line {line_num}.")    
        if not isinstance(anchor["text"], str):
            raise ValueError(f"'anchor.text' must be a str on line {line_num}.")    
        if not isinstance(anchor["bytes"], list) or not all(isinstance(x, int) for x in anchor["bytes"]):
            raise ValueError(f"'anchor.bytes' must be a list[int] on line {line_num}.")
        
        positives = record["positives"]
        if not isinstance(positives, list):
            raise ValueError(f"'positives' must be a list on line {line_num}.")
        
        for i, pos in enumerate(positives):
            if not isinstance(pos dict):
                raise ValueError(f"'positives[{i}]' must be a dict on line {line_num}.")
            
            required_pos = {"text", "bytes", "type", "source"}
            missing_pos = required_pos - set(pos.keys())
            if missing_pos:
                raise ValueError(f"'positives[{i}]' is missing keys {missing_pos} on line {line_num}.")
            
            if not isinstance(pos["text"], str):
                raise ValueError(f"'positives[{i}].text' must be a str on line {line_num}.")    
            if not isinstance(pos["bytes"], list) or not all(isinstance(x, int) for x in pos["bytes"]):
                raise ValueError(f"'positives[{i}].bytes' must be a list[int] on line {line_num}.")
            if pos["type"] not in {"phonetic", "script", "combined"}:
                raise ValueError(f"'positives[{i}].type' invalid on line {line_num}: {pos['type']}")
            if pos["source"] not in {"llm", "wikidata"}:
                raise ValueError(f"'positives[{i}].source' invalid on line {line_num}: {pos['source']}")
                
    
    