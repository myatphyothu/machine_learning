import numpy as np


def to_python_primitive_type(val):
    if type(val) in [np.float32, np.float64]:
        return float(val)
    elif type(val) in [np.int32, np.int64]:
        return int(val)
    else:
        return val
