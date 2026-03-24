import sys
import numpy as np
import utils as ut

def update_sor(tMesh, fixed, nx, ny, tolerance, Bi, t_Inf, omega, neumann_top, neumann_bottom, neumann_left, neumann_right) :
    
    iterations = 0
    max_iterations = 100000
    error = 0
    converged = False
    
    
    while not converged and iterations < max_iterations :
        tMesh_old = np.copy(tMesh)
        for j in range(1, ny-1) :
            for i in range(1, nx-1) :
                if not fixed[j, i] :
                    T_GS = 0.25 * (
                        tMesh[j+1, i] + 
                        tMesh[j-1, i] + 
                        tMesh[j, i+1] + 
                        tMesh[j, i-1]
                    )

                    tMesh[j, i] = (omega * T_GS) + ((1 - omega) * tMesh[j, i])

        ## Neumann Boundary Conditions ##
        if neumann_bottom :
            for j in range(1, nx-1) :
                T_GS = (
                    tMesh[0, j+1] +
                    tMesh[0, j-1] +
                    tMesh[1, j] +
                    Bi * t_Inf
                ) / (3 + Bi)

                tMesh[0, j] = (omega * T_GS) + ((1 - omega) * tMesh[0, j])
                
        if neumann_top :
            for j in range(1, nx-1) :
                T_GS = (
                    tMesh[ny-1, j+1] +
                    tMesh[ny-1, j-1] +
                    tMesh[ny-2, j] +
                    Bi * t_Inf
                ) / (3 + Bi)

                tMesh[ny-1, j] = (omega * T_GS) + ((1 - omega) * tMesh[ny-1, j])

        if neumann_left :
            for i in range(1, ny-1) :
                T_GS = (
                    tMesh[i+1, 0] +
                    tMesh[i-1, 0] +
                    tMesh[i, 1] +
                    Bi * t_Inf
                ) / (3 + Bi)

                tMesh[i, 0] = (omega * T_GS) + ((1 - omega) * tMesh[i, 0])  

        if neumann_right :
            for i in range(1, ny-1) :
                T_GS = (
                    tMesh[i+1, nx-1] +
                    tMesh[i-1, nx-1] +
                    tMesh[i, nx-2] +
                    Bi * t_Inf
                ) / (3 + Bi)

                tMesh[i, nx-1] = (omega * T_GS) + ((1 - omega) * tMesh[i, nx-1])

        if neumann_bottom and neumann_left :
            T_GS = (
                tMesh[1, 0] +
                tMesh[0, 1] +
                2 *Bi * t_Inf
            ) / (2 + (2 *Bi))

            tMesh[0, 0] = (omega * T_GS) + ((1 - omega) * tMesh[0, 0])

        if neumann_bottom and neumann_right :
            T_GS = (
                tMesh[1, nx-1] +
                tMesh[0, nx-2] +
                2 *Bi * t_Inf
            ) / (2 + (2 *Bi))

            tMesh[0, nx-1] = (omega * T_GS) + ((1 - omega) * tMesh[0, nx-1])

        if neumann_top and neumann_left :
            T_GS = (
                tMesh[ny-2, 0] +
                tMesh[ny-1, 1] +
                2 *Bi * t_Inf
            ) / (2 + (2 *Bi))

            tMesh[ny-1, 0] = (omega * T_GS) + ((1 - omega) * tMesh[ny-1, 0])

        if neumann_top and neumann_right :
            T_GS = (
                tMesh[ny-2, nx-1] +
                tMesh[ny-1, nx-2] +
                2 *Bi * t_Inf
            ) / (2 + (2 *Bi))

            tMesh[ny-1, nx-1] = (omega * T_GS) + ((1 - omega) * tMesh[ny-1, nx-1])

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
