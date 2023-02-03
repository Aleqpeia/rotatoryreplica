import time

import numpy as np
# from scipy.spatial.distance import pdist, squareform
import pyvista as pv

# import open3d as o3d

epsilon = 1e-10
# Noisy paralelepiped
x_min = -10
x_max = 10
y_min = -10
y_max = 10
z_min = -1
z_max = 1
noise_scale = 0.5
bump_height = 3
bump_width = 6
#x, y = np.mgrid[x_min:x_max:5j, y_min:y_max:5j]
#bump = if x, y < 1  bump_height * np.exp(-((x - 0.5 * (x_max - x_min))**2 + (y - 0.5 * (y_max - y_min))**2) / bump_width**2) else

def calc_direction(points, start_point, dist):
    xneighbors = np.abs(points[:, 0] - start_point[0]) < dist
    yneighbors = np.abs(points[:, 1] - start_point[1]) < dist
    zneighbors = np.abs(points[:, 2] - start_point[2]) < dist
    isneighbors = xneighbors * yneighbors * zneighbors
    neighbors = points[isneighbors]
    #print(neighbors)
    #print(np.sum(neighbors) / len(neighbors))
    #exit(0)
    close_enough = [neighbors[i] for i in range(len(neighbors))
                    if np.sum((neighbors[i] - start_point)**2) < dist*dist + epsilon]
    mean_point = np.mean(close_enough, axis=0)
    return np.array(mean_point)


def refine_surface(points, dist, n_iter, n_points):
    # random subset of points
    subset = points[np.random.choice(points.shape[0], n_points, replace=False)]
    # iterate over points and update position
    surface_points = []
    for i in range(n_points):
        point = subset[i]
        for j in range(n_iter):
            point = calc_direction(points, point, dist)
        surface_points.append(point)
    return surface_points

scale = 300
x = np.linspace(x_min, x_max, scale)
y = np.linspace(y_min, y_max, scale)
xs, ys = np.meshgrid(x, y)
xs += np.random.normal(scale=noise_scale, size=xs.shape)
ys += np.random.normal(scale=noise_scale, size=ys.shape)
@np.vectorize
def bump_function(x, y):
    xdenom = np.max([1 - x**2, 0.0000001])
    ydenom = np.max([1 - y**2, 0.0000001])
    denom = np.power(xdenom * ydenom, 2)
    return np.exp(1-(1/denom))
zs = bump_height*bump_function(xs/bump_width, ys/bump_width)
print(zs.shape)

zs += np.random.normal(scale=noise_scale, size=zs.shape)
grid = pv.StructuredGrid(xs, ys, zs)
# grid.plot()
plane = pv.Plane(center=(0,0,0), direction=(0,0,1), i_size=20, j_size=20, i_resolution=scale-1, j_resolution=scale-1)
print(plane.points.shape)
points = plane
# pv.plot(var_item=(points, grid.points), border=True, background='white', show_edges=True, show_bounds=True, show_axes=True)
#points_subset = grid.points[np.random.choice(grid.points.shape[0], 400, replace=False)]
t = time.time()
points_subset = refine_surface(grid.points, 0.5, 5, 300)
print("Took:", time.time()-t, "seconds.")

cloud = pv.PolyData(points_subset)

volume = cloud.delaunay_3d(alpha=2)

shell = volume.extract_geometry()
plotter = pv.Plotter()
#plotter.add_mesh(plane, color='blue', opacity=0.5)
plotter.add_mesh(shell, color='red', opacity=0.9)
plotter.background_color = "black"
plotter.add_points(grid.points, color='cyan', point_size=2, opacity=0.5)
plotter.show()


# Mean shif algorithm to regress the surface
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html

#