import numpy as np

def theoreticalSolution(x, y, Lx, Ly, tTop, tBottom, tLeft, tRight, terms = 100) :
    # Let t0 = tBottom for the calculation #
    T = 0 
    tFinal_Bottom = tBottom 
    tFinal_Top = 0
    tFinal_Left = 0
    tFinal_Right = 0

    tShift_top = tTop 

    for n in range(1, terms, 2) :

        # Top Layer #
        tFinal_Top += ((4 * (tTop - tBottom) / (n * np.pi)) * 
              np.sinh(n * np.pi * y / Lx) /
              np.sinh(n * np.pi * Ly / Lx) *
              np.sin(n * np.pi * x / Lx))
        
        # Left Layer #
        tFinal_Left += ((4 * (tLeft - tBottom) / (n * np.pi)) *
              np.sinh(n * np.pi * (Lx - x) / Ly) /
              np.sinh(n * np.pi * Lx / Ly) *
              np.sin(n * np.pi * y / Ly))
        
        # Right Layer #
        tFinal_Right += ((4 * (tRight - tBottom) / (n * np.pi)) *
              np.sinh(n * np.pi * x / Ly) /
              np.sinh(n * np.pi * Lx / Ly) *
              np.sin(n * np.pi * y / Ly))
        
    T = tFinal_Top + tFinal_Bottom + tFinal_Left + tFinal_Right
    return T


## Theoretical Solution Grid ##
def analyticalGrid(nx, ny, Lx, Ly, tTop, tBottom, tLeft, tRight) :
    tTheory = np.zeros((ny, nx))
    for i in range(0, ny) :
        for j in range (0, nx) :
            x = j / (nx - 1) * Lx
            y = i / (ny - 1) * Ly

            tTheory[i, j] = theoreticalSolution(x, y, Lx, Ly, tTop, tBottom, tLeft, tRight)
    
    return tTheory