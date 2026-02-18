# Re-export pipeline modules under clean names for importability.
# The numeric prefixes (02_, 03_, etc.) indicate execution order but are not
# valid Python identifiers, so we alias them here.

import importlib
import sys
from pathlib import Path

_scripts_dir = Path(__file__).parent

_aliases = {
    "filter_noncoding": "02_filter_noncoding",
    "build_negative_sets": "03_build_negative_sets",
    "annotate_regions": "04_annotate_regions",
    "extract_sequences": "05_extract_sequences",
}

for alias, filename in _aliases.items():
    _fqn = f"data.scripts.{alias}"
    if _fqn not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            _fqn,
            _scripts_dir / f"{filename}.py",
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[_fqn] = mod
            spec.loader.exec_module(mod)
