import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from llm_generator_pipeline.llm_client import call_ollama, parse_json_response
from llm_generator_pipeline.config import load_config