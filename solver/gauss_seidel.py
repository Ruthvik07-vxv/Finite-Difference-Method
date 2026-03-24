import numpy as np
from solver import utils as ut


def update_gs(tMesh, fixed, nx, ny, tolerance, Bi, t_Inf, neumann_top, neumann_bottom, neumann_left, neumann_right, verbose= True) :
    
    iterations = 0
    max_iterations = 100000
    error = 0
    converged = False
    
    while not converged and iterations < max_iterations :
        tMesh_old = np.copy(tMesh)
        for j in range(1, ny-1) :
            for i in range(1, nx-1) :
                if not fixed[j, i] :
                    tMesh[j, i] = 0.25 * (
                        tMesh[j+1, i] + 
                        tMesh[j-1, i] + 
                        tMesh[j, i+1] + 
                        tMesh[j, i-1]
                    )

        ## Neumann Boundary Conditions ##
        if neumann_bottom :
            for j in range(1, nx-1) :
                tMesh[0, j] = (
                    tMesh[0, j+1] +
                    tMesh[0, j-1] +
                    tMesh[1, j] +
                    Bi * t_Inf
                ) / (3 + Bi)

        if neumann_top :
            for j in range(1, nx-1) :
                tMesh[ny-1, j] = (
                    tMesh[ny-1, j+1] +
                    tMesh[ny-1, j-1] +
                    tMesh[ny-2, j] +
                    Bi * t_Inf
                ) / (3 + Bi)

        if neumann_left :
            for i in range(1, ny-1) :
                tMesh[i, 0] = (
                    tMesh[i+1, 0] +
                    tMesh[i-1, 0] +
                    tMesh[i, 1] +
                    Bi * t_Inf
                ) / (3 + Bi)

        if neumann_right :
            for i in range(1, ny-1) :
                tMesh[i, nx-1] = (
                    tMesh[i+1, nx-1] +
                    tMesh[i-1, nx-1] +
                    tMesh[i, nx-2] +
                    Bi * t_Inf
                ) / (3 + Bi)

        if neumann_bottom and neumann_left :
            tMesh[0, 0] = (
                tMesh[1, 0] +
                tMesh[0, 1] +
                2 *Bi * t_Inf
            ) / (2 + (2 *Bi))

        if neumann_bottom and neumann_right :
            tMesh[0, nx-1] = (
                tMesh[1, nx-1] +
                tMesh[0, nx-2] +
                2 *Bi * t_Inf
            ) / (2 + (2 *Bi))

        if neumann_top and neumann_left :
            tMesh[ny-1, 0] = (
                tMesh[ny-2, 0] +
                tMesh[ny-1, 1] +
                2 *Bi * t_Inf
            ) / (2 + (2 *Bi))

        if neumann_top and neumann_right :
            tMesh[ny-1, nx-1] = (
                tMesh[ny-2, nx-1] +
                tMesh[ny-1, nx-2] +
                2 *Bi * t_Inf
            ) / (2 + (2 *Bi))

        converged, error = ut.convergence_check(tMesh, tMesh_old, tolerance)
        iterations += 1

        if verbose and iterations % 500 == 0 :
            print()
            print("Number of iterations: ", iterations)
            print("Converged error: ", error)
            print("Running status: true")


    if converged :
        print()
        print("Convergence achieved")
        print("Number of iterations taken for convergence: ", iterations)
        print("Converged error: ", error)
        

    elif iterations >= max_iterations :
        print()
        print("Maximum iterations reached without convergence.")
        print("Possible converged error: ", error)
        print("Tolerace: ", tolerance)
        print("Consider adjusting the tolerance to achieve convergence.")

    return tMesh

def update_gs_heat_generation(tMesh, fixed, nx, ny, tolerance, length, height, q, k, h, t_Inf, neumann_top, neumann_bottom, neumann_left, neumann_right) :
    dx = length / (nx - 1)
    dy = height / (ny - 1)

    Bi_x = h * dx / k
    Bi_y = h * dy / k
    
    iterations = 0
    max_iterations = 100000
    error = 0
    converged = False
    
    while not converged and iterations < max_iterations :
        tMesh_old = np.copy(tMesh)
        for j in range(1, ny-1) :
            for i in range(1, nx-1) :
                if not fixed[j, i] :
                    tMesh[j, i] = (((tMesh[j+1, i] + tMesh[j-1, i]) / dx**2) + 
                                   ((tMesh[j, i+1] + tMesh[j, i-1]) / dy**2) + 
                                   (q/k)) / ((2 / dx**2) + (2 / dy**2))

        ## Neumann Boundary Conditions ##
        if neumann_bottom :
            for j in range(1, nx-1) :
                tMesh[0, j] = (
                    (tMesh[0, j+1] / dx**2) +
                    (tMesh[0, j-1] / dx**2) +
                    (tMesh[1, j] / dy**2 ) +
                    ((Bi_y * t_Inf) / dy**2)
                ) / ((Bi_y / dy**2) + (1 / dy**2) + (2 / dx**2))

        if neumann_top :
            for j in range(1, nx-1) :
                tMesh[ny-1, j] = (
                    (tMesh[ny-1, j+1] / dx**2) +
                    (tMesh[ny-1, j-1] / dx**2) +
                    (tMesh[ny-2, j] / dy**2) +
                    ((Bi_y * t_Inf) / dy**2)
                ) / ((Bi_y / dy**2) + (1 / dy**2) + (2 / dx**2))

        if neumann_left :
            for i in range(1, ny-1) :
                tMesh[i, 0] = (
                    (tMesh[i+1, 0] / dy**2) +
                    (tMesh[i-1, 0] / dy**2) +
                    (tMesh[i, 1] / dx**2) +
                    ((Bi_x * t_Inf) / dx**2)
                ) / ((Bi_x / dx**2) + (1 / dy**2) + (2 / dx**2))

        if neumann_right :
            for i in range(1, ny-1) :
                tMesh[i, nx-1] = (
                    (tMesh[i+1, nx-1] / dy**2) +
                    (tMesh[i-1, nx-1] / dy**2) +
                    (tMesh[i, nx-2] / dx**2) +
                    ((Bi_x * t_Inf) / dx**2)
                ) / ((Bi_x / dx**2) + (1 / dy**2) + (2 / dx**2))

        if neumann_bottom and neumann_left :
            tMesh[0, 0] = (
                (tMesh[1, 0] * dx**2) +
                (tMesh[0, 1] * dy**2) +
                (Bi_x * t_Inf * dy**2) +
                (Bi_y * t_Inf * dx**2)
            ) / (dx**2 + dy**2 + (Bi_x * dy**2) + (Bi_y * dx**2))

        if neumann_bottom and neumann_right :
            tMesh[0, nx-1] = (
                (tMesh[1, nx-1] * dx**2) +
                (tMesh[0, nx-2] * dy**2) +
                (Bi_x * t_Inf * dy**2) +
                (Bi_y * t_Inf * dx**2)
            ) / (dx**2 + dy**2 + (Bi_x * dy**2) + (Bi_y * dx**2))

        if neumann_top and neumann_left :
            tMesh[ny-1, 0] = (
                (tMesh[ny-2, 0] * dy**2) +
                (tMesh[ny-1, 1] * dx**2) +
                (Bi_x * t_Inf * dy**2) +
                (Bi_y * t_Inf * dx**2)
            ) / (dx**2 + dy**2 + (Bi_x * dy**2) + (Bi_y * dx**2))

        if neumann_top and neumann_right :
            tMesh[ny-1, nx-1] = (
                (tMesh[ny-2, nx-1] * dy**2) +
                (tMesh[ny-1, nx-2] * dx**2) +
                (Bi_x * t_Inf * dy**2) +
                (Bi_y * t_Inf * dx**2)
            ) / (dx**2 + dy**2 + (Bi_x * dy**2) + (Bi_y * dx**2))

        converged, error = ut.convergence_check(tMesh, tMesh_old, tolerance)
        iterations += 1

        if iterations % 500 == 0 :
            print()
            print("Number of iterations: ", iterations)
            print("Converged error: ", error)
            print("Running status: true")


    if converged :
        print()
        print("Convergence achieved")
        print("Number of iterations taken for convergence: ", iterations)
        print("Converged error: ", error)
        

    elif iterations >= max_iterations :
        print()
        print("Maximum iterations reached without convergence.")
        print("Possible converged error: ", error)
        print("Tolerace: ", tolerance)
        print("Consider adjusting the tolerance to achieve convergence.")

    return tMesh
