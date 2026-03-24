#Standard library imports
import sys
import os
import multiprocessing as mp
import time

#Third party imports
import numpy as np
import matplotlib.pyplot as plt

#Local imports
import mesh
import boundary as bd
import postprocess as pp

from solver import jacobi as jacobi_solver
from solver import sor as sor_solver
from solver import gauss_seidel as gs_solver
from solver import analytical as th_solver
from solver import utils as ut  

def run_solver(method, tMesh, fixed, nx, ny, tolerance, Bi, t_Inf, omega, neumann_top, neumann_bottom, neumann_left, neumann_right) :
    tMesh_local = np.copy(tMesh)
    start = time.time()
    if method == "jacobi" :
        result = jacobi_solver.update_jacobi(tMesh_local, fixed, nx, ny, tolerance, Bi, t_Inf, neumann_top, neumann_bottom, neumann_left, neumann_right, verbose= False)
    elif method == "gauss-seidel" :
        result = gs_solver.update_gs(tMesh_local, fixed, nx, ny, tolerance, Bi, t_Inf, neumann_top, neumann_bottom, neumann_left, neumann_right, verbose= False)
    elif method == "sor" :
        result = sor_solver.update_sor(tMesh_local, fixed, nx, ny, tolerance, Bi, t_Inf, omega, neumann_top, neumann_bottom, neumann_left, neumann_right, verbose= False)
    end = time.time()
    return method, result, end - start

def main() :
    length = float(input("Enter the length of the domain: "))
    width = float(input("Enter the width of the domain: "))
    int_gen = input("Is there internal heat generation present (q)? (y/n): ").lower()
    if int_gen != 'y' and int_gen != 'n' :
        print("Invalid choice, please try again!")
        sys.exit()
    nx = int(input("Enter the number of nodes in x direction(-1 to auto assign): "))
    if nx == -1 :
        nx = 40
    if length != width :
        ny = int(input("Enter the number of nodes in y direction(-1 to auto assign): "))
        if ny == -1 :
            ny = 40
    else :
        ny = nx

    dx = length / (nx - 1)
    dy = width / (ny - 1)

    print()

    tTop = float(input("Enter the temperature of the Top surface(-1 if not specified): "))
    if tTop == -1 :
        tTop = 0
    if tTop <= -273.15 :
        print("Error in temperature, cannot be less than -273.15 in celsius")
        sys.exit()
    tBottom = float(input("Enter the temperature of the Bottom surface(-1 if not specified): "))
    if tBottom == -1 :
        tBottom = 0
    if tBottom <= -273.15 :
        print("Error in temperature, cannot be less than -273.15 in celsius")
        sys.exit()
    tLeft = float(input("Enter the temperature of the Left surface(-1 if not specified): "))
    if tLeft == -1 :
        tLeft = 0
    if tLeft <= -273.15 :
        print("Error in temperature, cannot be less than -273.15 in celsius")
        sys.exit()
    tRight = float(input("Enter the temperature of the Right surface(-1 if not specified): "))
    if tRight == -1 :
        tRight = 0
    if tRight <= -273.15 :
        print("Error in temperature, cannot be less than -273.15 in celsius")
        sys.exit()
    tolerance = float(input("Enter the Convergence Error: "))
    
    tMesh, fixed = mesh.CreateMesh(nx, ny, tTop, tBottom, tLeft, tRight)

    conv = input("Is there convection present at any of the boundaries? (y/n): ").lower()
    if conv == 'y' :
        h, k, t_Inf, neumann_top, neumann_bottom, neumann_left, neumann_right, fixed = bd.check_convection_boundaries(fixed)
    elif conv == 'n' :
        h = 0
        k = 1
        t_Inf = 0
        neumann_top = False
        neumann_bottom = False
        neumann_left = False
        neumann_right = False
    else :
        print("Wrong input! Try again")
        return main()
    
    Bi_x, Bi_y = ut.biot_number(h, k, length, width, tMesh)
    Bi = (Bi_x + Bi_y) / 2 

    folder_name = "results"
    if not os.path.exists(folder_name) :
        os.makedirs(folder_name)

    print()
    if int_gen == 'y' :
        q = float(input("Enter the internal heat generation rate q: "))
        k_mat = float(input("Enter the thermal conductivity k: "))
        print("Running Gauss-Seidel (Poisson) solver...")
        start = time.time()
        tMesh = gs_solver.update_gs_heat_generation(tMesh, fixed, nx, ny, tolerance, length, width, q, k_mat, h, t_Inf, neumann_top, neumann_bottom, neumann_left, neumann_right)
        duration = time.time() - start
        print()
        print("Benchmark Results (Internal Heat Generation)")
        print("-----------------------------------------------------")
        print(f"{'Method':20s}{'Time (s)':>12s}")
        print(f"{'Gauss-Seidel (Poisson)':20s}{duration:12.4f}")
    else :
        print("Select the solver to use: ")
        print("1. Jacobi")
        print("2. Gauss-Seidel")
        print("3. Successive Over-Relaxation (SOR)")
        print("4. Run all solvers and compare results")
        solver_choice = int(input("Enter the number corresponding to the solver: "))
        if solver_choice == 1 :
            start = time.time()
            tMesh = jacobi_solver.update_jacobi(tMesh, fixed, nx, ny, tolerance, Bi, t_Inf, neumann_top, neumann_bottom, neumann_left, neumann_right)
            duration = time.time() - start
            print()
            print("Benchmark Results")
            print("-----------------------------------------------------")
            print(f"{'Method':20s}{'Time (s)':>12s}")
            print(f"{'Jacobi':20s}{duration:12.4f}")
        elif solver_choice == 2 :
            start = time.time()
            tMesh = gs_solver.update_gs(tMesh, fixed, nx, ny, tolerance, Bi, t_Inf, neumann_top, neumann_bottom, neumann_left, neumann_right)
            duration = time.time() - start
            print()
            print("Benchmark Results")
            print("-----------------------------------------------------")
            print(f"{'Method':20s}{'Time (s)':>12s}")
            print(f"{'Gauss-Seidel':20s}{duration:12.4f}")
        elif solver_choice == 3 :
            omega = float(input("Enter the relaxation factor (0 < omega < 2): "))
            if omega <= 0 or omega >= 2 :
                print("Invalid relaxation factor! Please enter a value between 0 and 2.")
                sys.exit(1)
            start = time.time()
            tMesh = sor_solver.update_sor(tMesh, fixed, nx, ny, tolerance, Bi, t_Inf, omega, neumann_top, neumann_bottom, neumann_left, neumann_right)
            duration = time.time() - start
            print()
            print("Benchmark Results")
            print("-----------------------------------------------------")
            print(f"{'Method':20s}{'Time (s)':>12s}")
            print(f"{'SOR':20s}{duration:12.4f}")
        elif solver_choice == 4 :
            omega = float(input("Enter the relaxation factor (0 < omega < 2): "))
            if omega <= 0 or omega >= 2 :
                print("Invalid relaxation factor! Please enter a value between 0 and 2.")
                sys.exit(1)
            print("Running all solvers in parallel...")
            methods = ["jacobi", "gauss-seidel", "sor"]
            with mp.Pool(processes=3) as pool :
                results = pool.starmap(run_solver, [(method, tMesh, fixed, nx, ny, tolerance, Bi, t_Inf, omega, neumann_top, neumann_bottom, neumann_left, neumann_right) for method in methods])
            print()
            print("Benchmark Results")
            print("----------------------------------------------------------------")
            print(f"{'Method':20s}{'Time (s)':>12s}")
            for method, result, duration in results :
                print(f"{method.capitalize():20s}{duration:12.4f}")
                if method == "jacobi" :
                    tMesh_jacobi = result
                    file_path_jacobi_1 = os.path.join(folder_name, "jacobi_temperature_contour.txt")
                    file_path_jacobi_2 = os.path.join(folder_name, "jacobi_temperature_contour.csv")
                    pp.saveTemperatureGrid(tMesh_jacobi, file_path_jacobi_1)
                    pp.save_CSV(tMesh_jacobi, file_path_jacobi_2)
                elif method == "gauss-seidel" :
                    tMesh_gs = result
                    file_path_gs_1 = os.path.join(folder_name, "gauss-seidel_temperature_contour.txt")
                    file_path_gs_2 = os.path.join(folder_name, "gauss-seidel_temperature_contour.csv")
                    pp.saveTemperatureGrid(tMesh_gs, file_path_gs_1)
                    pp.save_CSV(tMesh_gs, file_path_gs_2)
                elif method == "sor" :
                    tMesh_sor = result
                    file_path_sor_1 = os.path.join(folder_name, "sor_temperature_contour.txt")
                    file_path_sor_2 = os.path.join(folder_name, "sor_temperature_contour.csv")
                    pp.saveTemperatureGrid(tMesh_sor, file_path_sor_1)
                    pp.save_CSV(tMesh_sor, file_path_sor_2)
            best = min(results, key=lambda x: x[2])
            print("----------------------------------------------------------------")
            print(f"The fastest solver is: {best[0].capitalize()} with a time of {best[2]:.4f} seconds.")
            tMesh = best[1] 
        else :
            print("Wrong input! Try again")
            return main()
    
    file_path = os.path.join(folder_name, "temperature_contour.png")
    pp.plot_temperature_contours(tMesh, title="Temperature Contour", save_path=file_path)

    file_path_2 = os.path.join(folder_name, "temperature_isotherms.png")
    pp.plot_temperature(tMesh, title="Temperature Isotherms", save_path=file_path_2)

    file_path_3 = os.path.join(folder_name, "Combined_Graph.png")
    pp.plot_combined_graph(tMesh, title="Combined Graph", save_path=file_path_3)

    file_path_4 = os.path.join(folder_name, "Temperature_calculated.txt")
    pp.saveTemperatureGrid(tMesh, file_path_4)

    file_path_5 = os.path.join(folder_name, "Temperature_calculated.csv")
    pp.save_CSV(tMesh, file_path_5)

    if conv != 'y' and int_gen != 'y' :
        print()
        print()
        print("Analytical Solution: ")
        tAnalytical = th_solver.analyticalGrid(nx, ny, length, width, tTop, tBottom, tLeft, tRight)
        print("Theoretical temperatures are calculated successfully.")

        file_path_6 = os.path.join(folder_name, "analytical_contours.png")
        pp.plot_temperature_contours(tAnalytical, title="Analytical Temperature Contours", save_path=file_path_6)

        file_path_7 = os.path.join(folder_name, "analytical_isotherms.png")
        pp.plot_temperature(tAnalytical, title="Analytical Temperature Isotherms", save_path=file_path_7)

        file_path_8 = os.path.join(folder_name, "temperature_analytical.txt")
        pp.saveTemperatureGrid(tAnalytical, file_path_8)

        file_path_9 = os.path.join(folder_name, "temperature_analytical.csv")
        pp.save_CSV(tAnalytical, file_path_9)

        errorCalc = ut.calculate_error(tMesh, tAnalytical)
        max_error = np.max(errorCalc)
        print("Maximum error between numerical and analytical solution: ", max_error)

        file_path_10 = os.path.join(folder_name, "error_differences.png")
        pp.plot_temperature_contours(errorCalc, title="Error Differences", save_path=file_path_10)

        file_path_11 = os.path.join(folder_name, "error_differences.txt")
        pp.saveTemperatureGrid(errorCalc, file_path_11)

        file_path_12 = os.path.join(folder_name, "error_differences.csv")
        pp.save_CSV(errorCalc, file_path_12)

        temp_check = (input("Do you want to check the temperature at a specific point? (y/n): ").lower())
        while temp_check == 'y' :
            x = int(input(f"Enter the value of x (< {nx}): "))
            if x < 0 or x >= nx :
                print("Invalid x value! Try again.")
                continue
            y = int(input(f"Enter the value of y (< {ny}): "))
            if y < 0 or y >= ny :
                print("Invalid y value! Try again.")
                continue
            print(f"Analytical Solution: {tAnalytical[y, x]}")
            print(f"Converged Solution: {tMesh[y, x]}")
            print(f"Error in Solution: {errorCalc[y, x]}")
            temp_check = (input("Do you want to check another point? (y/n): ").lower())
            if temp_check != 'y' :
                break
            else :
                continue
    else :
        print("Analytical solution is not available for convection boundary conditions.")
        temp_check = (input("Do you want to check the temperature at a specific point? (y/n): ").lower())
        while temp_check == 'y' :
            x = int(input(f"Enter the value of x (< {nx}): "))
            if x < 0 or x >= nx :
                print("Invalid x value! Try again.")
                continue
            y = int(input(f"Enter the value of y (< {ny}): "))
            if y < 0 or y >= ny :
                print("Invalid y value! Try again.")
                continue
            print(f"Converged Solution: {tMesh[y, x]}")
            temp_check = (input("Do you want to check another point? (y/n): ").lower())
            if temp_check != 'y' :
                break
            else :
                continue

if __name__ == "__main__" :
    main()
