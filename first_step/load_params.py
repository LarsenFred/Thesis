import numpy as np
from pathlib import Path

def load_params_txt(path):
    """
    Expects a two‐column file with
      name[TAB]value
    where value is either a float (e.g. “0.2732”) or
    a space‐delimited vector in square brackets (e.g. “[0.4 0.98 1.53 0.85]”).
    Returns a dict with floats or numpy.ndarray.
    """
    params = {}
    with open(path, "r") as f:
        header = next(f)            # skip header line
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split only on first whitespace (or tab)
            key, val = line.split(None, 1)
            # detect vectors
            if val.startswith("[") and val.endswith("]"):
                vals = val[1:-1].strip()        # drop [ ]
                # parse space‐separated numbers
                arr = np.fromstring(vals, sep=" ")
                params[key] = arr
            else:
                params[key] = float(val)
    return params
