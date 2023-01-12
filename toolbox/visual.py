import MDAnalysis as md
import MDAnalysis.core.universe
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d import *

import procfunc as pf

# Load the trajectory file
u = md.core.universe.Universe(sys.argv[0], sys.argv[1], in_memory=True)
tracker = sys.argv[3] # picks certain type of residues/atoms/molecules for atomgroup selection, in this case upper leaflet of membrane
def streamplot(u, tracker):

    leaflets = pf.leaflet(u, tracker)
    for lipids in leaflets[:,:]:
        interest = u.select_atoms(lipids)
        mem = md.coordinates.memory.MemoryReader(interest)





    # Define the timestep for the calculation (e.g. 1 ns)
    dt = 1 * mem.trajectory.dt

    # Initialize empty arrays to store the positions and velocities
    positions = []
    velocities = []

    # Loop over all frames in the trajectory
    for ts in mem.trajectory:
        # Append the positions and velocities of the atoms in the leaflet
        positions.append(mem.coordinate_array)
        velocities.append(mem.velocity_array)

        # Convert the positions and velocities to numpy arrays
        positions = np.array(positions)
        velocities = np.array(velocities)

        # Select only x and y component of velocities
        velocities_2D = velocities[:,:,:2]

        # Calculate the flow of lipids in the membrane
        flow_x = velocities_2D[:,:,0]
        flow_y = velocities_2D[:,:,1]

        # Plot the 2D flow as a stream plot
        x = positions[0][:,0]
        y = positions[0][:,1]
        plt.streamplot(x, y, flow_x, flow_y)
        plt.xlabel("X (A)")
        plt.ylabel("Y (A)")
        plt.show()

def memb_surface(points, normals):
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([pcd])
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
        #mesh_out = mesh.filter_smooth_simple(number_of_iterations=5)
        print([mesh])

        # mesh.vertex_colors = o3d.utility.Vector3dVector(densities[:, None])
        # colormap = o3d.pipelines.color_map.
        # min_density = np.min(densities)
        # max_density = np.max(densities)
        # mesh.paint_uniform_color(color_map.map_to_color(mesh.vertex_colors))
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)