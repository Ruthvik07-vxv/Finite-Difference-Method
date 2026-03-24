import numpy as np
import matplotlib.pyplot as plt
import time

def create_plot_grid(tMesh) :
    ny, nx = tMesh.shape
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y

def plot_temperature(tMesh, title="Temperature Isotherms", save_path=None):
    X, Y = create_plot_grid(tMesh)

    plt.figure(figsize=(8, 6))

    contour = plt.contourf(X, Y, tMesh, levels=200, cmap='inferno')
    plt.colorbar(contour)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")

    plt.show()
    time.sleep(0.5)
    plt.close()

def plot_temperature_contours(tMesh, title="Temperature Contour", save_path=None):
    X, Y = create_plot_grid(tMesh)

    plt.figure(figsize=(8, 6))

    contour = plt.contour(X, Y, tMesh, levels=20, cmap='inferno')
    plt.clabel(contour, inline=True, fontsize=8)

    plt.colorbar(contour)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")

    plt.show()
    time.sleep(0.5)
    plt.close()

def plot_combined_graph(tMesh, title="Temperature Distribution with Contours", save_path=None):
    X, Y = create_plot_grid(tMesh)

    plt.figure(figsize=(8, 6))

    contour_filled = plt.contourf(X, Y, tMesh, levels=200, cmap='inferno')
    contour_lines = plt.contour(X, Y, tMesh, levels=20, colors='white', linewidths=0.5)

    plt.clabel(contour_lines, inline=True, fontsize=8)
    plt.colorbar(contour_filled)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")

    plt.show()
    time.sleep(0.5)
    plt.close()

def saveTemperatureGrid(tMesh, filename, flip=True):
    data = np.flipud(tMesh) if flip else tMesh

    with open(filename, "w") as f:
        for row in data:
            for val in row:
                f.write(f"{val:10.4f} ")
            f.write("\n")

    print(f"Temperature Grid saved to {filename}")

def save_CSV(tMesh, filename, flip=True) :
    data = np.flipud(tMesh) if flip else tMesh
    np.savetxt(filename, data, delimiter=",", fmt="%.4f")
    print(f"Temperature Grid saved to {filename}")

