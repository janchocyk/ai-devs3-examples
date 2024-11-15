import codecs
import os
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)


def os_env_mandatory(env_variable_name: str) -> Any:
    env_variable = codecs.decode(os.getenv(env_variable_name, None), "unicode_escape")
    if env_variable is None or env_variable == "":
        raise RuntimeError(
            f"Environment variable '{env_variable_name}' is not set, but is mandatory."
        )
    return env_variable


class Config:
    APP_DIR = Path(__file__).resolve()
    QDRANT_STORAGE = APP_DIR / "vectorstore"
    MEMORIES = APP_DIR / "memories_contener"
    

config = Config()