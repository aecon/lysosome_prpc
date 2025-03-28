"""
Plate layout
"""

import numpy as np


Nrows = 8
Ncolumns = 12

row_letters = np.asarray(["A", "B", "C", "D", "E", "F", "G", "H"])

def well2id(w):
    idx = np.where(row_letters == w)[0][0]
    return idx



