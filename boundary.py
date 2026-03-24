import numpy as np

def check_boundary_conditions() :
    print("Enter boundary conditions for the mesh: ")
    tTop = float(input("Enter the temperature at the top boundary: "))
    tBottom = float(input("Enter the temperature at the bottom boundary: "))
    tLeft = float(input("Enter the temperature at the left boundary: "))
    tRight = float(input("Enter the temperature at the right boundary: "))

    return tTop, tBottom, tLeft, tRight

def convective_layer_top() :
    neumann_top = input("Is there a convective layer at the top boundary? (y/n): ").lower()
    if neumann_top == 'y' :
        neumann_top = True
    elif neumann_top == 'n' :
        neumann_top = False
    else :
        print("Wrong input! Try again")
        return convective_layer_top()
    
    return neumann_top
    
def convective_layer_bottom() :
    neumann_bottom = input("Is there a convective layer at the bottom boundary? (y/n): ").lower()
    if neumann_bottom == 'y' :
        neumann_bottom = True
    elif neumann_bottom == 'n' :
        neumann_bottom = False
    else :
        print("Wrong input! Try again")
        return convective_layer_bottom()
    
    return neumann_bottom
    
def convective_layer_left() :
    neumann_left = input("Is there a convective layer at the left boundary? (y/n): ").lower()
    if neumann_left == 'y' :
        neumann_left = True
    elif neumann_left == 'n' :
        neumann_left = False
    else :
        print("Wrong input! Try again")
        return convective_layer_left()
    
    return neumann_left
    
def convective_layer_right() :
    neumann_right = input("Is there a convective layer at the right boundary? (y/n): ").lower()
    if neumann_right == 'y' :
        neumann_right = True
    elif neumann_right == 'n' :
        neumann_right = False
    else :
        print("Wrong input! Try again")
        return convective_layer_right()

    return neumann_right

def check_convection_boundaries(fixed) :
    conv = input("Is convection present at any of the boundaries? (y/n): ").lower()
    if conv == 'n' :
        return None, None, None, False, False, False, False, fixed
    
    elif conv != 'y' and conv != 'n' :
        print("Wrong input! Try again")
        return check_convection_boundaries(fixed)
    
    else :
        h = float(input("Enter the convective heat transfer coefficient: "))
        t_inf = float(input("Enter the ambient temperature: "))
        k = float(input("Enter the thermal conductivity of the material: "))
        neumann_top = convective_layer_top()
        neumann_bottom = convective_layer_bottom()
        neumann_left = convective_layer_left()
        neumann_right = convective_layer_right()

        fixed[0, :] = not neumann_bottom
        fixed[-1, :] = not neumann_top
        fixed[:, 0] = not neumann_left
        fixed[:, -1] = not neumann_right

        return h, k, t_inf, neumann_top, neumann_bottom, neumann_left, neumann_right, fixed

