import numpy as np

def CreateMesh(nx, ny, tTop = 0, tBottom = 0, tLeft = 0, tRight = 0) :
    ##  Creating a mesh of nx, ny  ##
    tMesh = np.zeros((ny, nx), dtype=float)

    ## Dirichlet Conditions ##
    tMesh[0, :] = tBottom
    tMesh[ny-1, :] = tTop
    tMesh[:, 0] = tLeft
    tMesh[:, nx-1] = tRight

    ##  Creating a fixed temperature Node  ##
    fixed = np.zeros((ny, nx), dtype=bool)

    fixed[0, :] = True
    fixed[ny-1, :] = True
    fixed[:, 0] = True
    fixed[:, nx-1] = True
    print("Mesh Created Successfully!")

    return tMesh, fixed