import os
import shutil
from typing import Tuple

def get_name(path: str) -> Tuple[str, str]:
    filename = os.path.basename(path)
    wo_ext = ".".join(filename.split(".")[:-1])
    return filename, wo_ext

def create_dir(path: os.PathLike, del_old: bool=False):
    if del_old and os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=not del_old)