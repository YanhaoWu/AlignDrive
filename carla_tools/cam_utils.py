"""
    Copyright (C) 2022 Martin Ahrnbom
    This work is released under the MIT License. 
    See the file LICENSE for details


    Utility functions
"""

from math import sqrt
from typing import List
import numpy as np 
import carla
import io 

def loc_dist(a, b):
    return sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def vector_normalize(v:carla.Vector3D):
    norm = v.x**2 + v.y**2 + v.z**2
    new = carla.Vector3D(x=v.x/norm, y=v.y/norm, z=v.z/norm)
    return new 

def vector_from_to(a:carla.Vector3D, b:carla.Vector3D):
    dx = b.x - a.x
    dy = b.y - a.y
    dz = b.z - a.z
    return carla.Vector3D(dx, dy, dz)

def scalar_product(a:carla.Vector3D, b:carla.Vector3D):
    return a.x*b.x + a.y*b.y + a.z*b.z

def vector_dist(a, b):
    return np.linalg.norm(a-b)

def normalize_numpy_vector(x: np.ndarray):
    n = np.linalg.norm(x)
    if n > 0.00001:
        return x / n 
    else:
        return None 

# long_str(2) -> '0002'
# long_str(42, 3) -> '042'
def long_str(i:int, N:int=4, padding='0'):
    s = str(i)
    n = len(s)
    if n < N:
        s = padding*(N-n) + s 
    
    return s 

# Removes 'intro' from left part of 'text', raises error if not found
def good_lstrip(text, intro):
    assert(len(intro) <= len(text))
    l = len(intro)
    first = text[:l]
    assert(first == intro)

    return text[l:]

def intr(x):
    return int(round(float(x)))

# Projective flattening, scales homogeneous coordinates so that last coordinate is always one
def pflat(x):
    if len(x.shape) == 1:
        x /= x[-1]
    else:
        x /= x[-1, :]
    return x

def print_table(row_names:List[str], col_names:List[str], matrix:np.ndarray,
                decimals=2):
    
    matrix = np.around(matrix, decimals=decimals)
    
    row_names = np.array(row_names, dtype=str).reshape((len(row_names), 1))
    matrix = np.hstack([row_names, matrix])
    col_names = np.array(['', *col_names], dtype=str)
    col_names = col_names.reshape((1, len(col_names)))
    matrix = np.vstack([col_names, matrix])

    max_len = max([len(v) for v in matrix.flatten()])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i,j]
            matrix[i, j] = long_str(val, max_len, padding=' ')

    print(np.array2string(matrix, max_line_width=200))
    