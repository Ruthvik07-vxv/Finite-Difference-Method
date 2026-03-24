import numpy as np

def convergence_check(tMesh, tMesh_old, tolerance) :
    error = np.max(np.abs(tMesh - tMesh_old))
    if error < tolerance :
        return True, error
    else :
        return False, error
    
def biot_number(h, k, length, height, tMesh) :
    dx = length / (tMesh.shape[1] - 1)
    dy = height / (tMesh.shape[0] - 1)

    Bi_x = h * dx / k
    Bi_y = h * dy / k

    return Bi_x, Bi_y

def calculate_error(tMesh, tTheory) :
    error = np.abs(tTheory - tMesh)
    return error