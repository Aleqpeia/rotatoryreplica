import MDAnalysis as mda
import numpy as np
from scipy.spatial import Delaunay
from sklearn.linear_model import LogisticRegressionCV

import procfunc as pf


# simple multivariate linear regression of the lipid packing anisotropy function that gives TRUE output if system undergoes phase separation and FALSE if it is not

def predict_phase_separation(traj, lipids):
    lipids = traj.select_atoms(lipids)

    S2_head = []
    S2_tail = []
    for lipid in lipids.residues:
        head = lipid.select_atoms("name PO4")
        tail = lipid.select_atoms("name C12 C13 C14 C15 C16")
        S2_head.append(head.principal_axes()[1][2])
        S2_tail.append(tail.principal_axes()[1][2])

    X = np.vstack([S2_head, S2_tail]).T
    model = LogisticRegressionCV().fit(X, np.zeros(len(X)))


    # Predict whether the system undergoes phase separation or not
    phase_separation = model.coef_[0] > model.coef_[1]
    return phase_separation
print()







def tessellate_surface(points, physical_param):
    """
    Tessellate a surface from a set of input points, using an assigned physical parameter for color mapping.
    Outputs the neighbor lists, vertices, and volumes of each triangle

    Parameters:
    - points (numpy array): the set of input points
    - physical_param (numpy array): the physical parameter assigned to each point
    """
    # Compute the Delaunay triangulation
    tri = Delaunay(points)
    triangles = tri.simplices
    print(len(triangles))
    # Compute the neighbor lists, vertices and volumes of each triangle
    # neighbor_lists = [tri.neighbors[i] for i in range(tri.neighbors.shape[0])]
    # vertices = [tri.vertices[i] for i in range(tri.vertices.shape[0])]
    # volumes = [np.abs(np.linalg.det(np.vstack((points[triangle],np.ones(3))))) for triangle in triangles]
    neighbor_lists = tri.neighbors
    vertices = tri.vertices

    # compute volume of each triangle
    # volumes = np.zeros(triangles.shape[0])
    plt.tripcolor(points[:,0], points[:,1], triangles, facecolors=physical_param)
    plt.show()
    return tri, triangles






def remove_overlapping_points(points):
    unique_points = []
    for point in points:
        if point not in unique_points:
            unique_points.append(point)
    return unique_points

# Surface reconstruction method using Poisson surface reconstruction algorithm with mesh output as base and densities output as colormap
def tessellate_surface(points, physical_param):
    """
    Tessellate a surface from a set of input points, using an assigned physical parameter for color mapping.
    Outputs the neighbor lists, vertices, and volumes of each triangle

    Parameters:
    - points (numpy array): the set of input points
    - physical_param (numpy array): the physical parameter assigned to each point
    """
    # Compute the Delaunay triangulation
    tri = Delaunay(points)
    triangles = tri.simplices
    print(len(triangles))
    # Compute the neighbor lists, vertices and volumes of each triangle
    # neighbor_lists = [tri.neighbors[i] for i in range(tri.neighbors.shape[0])]
    # vertices = [tri.vertices[i] for i in range(tri.vertices.shape[0])]
    # volumes = [np.abs(np.linalg.det(np.vstack((points[triangle],np.ones(3))))) for triangle in triangles]
    neighbor_lists = tri.neighbors
    vertices = tri.vertices

    # compute volume of each triangle
    # volumes = np.zeros(triangles.shape[0])
    # Plot the triangulation
    plt.tripcolor(points[:,0], points[:,1], triangles, facecolors=physical_param)
    plt.show()
    return tri, triangles











def preprocess(u):

    """Extract all the informations from the simulations files and return a class with the system
    Argument(s):
        coordinates_file {str} -- Relative path to the coordinates file of the system (e.g. .gro file).
        structure_file {str} -- Relative path to the structure file of the system (e.g. .tpr file).
        type {str} -- Name of the molecule type to extract the information from. The type should be similar as the one listed in getTypes().
        trj {str} -- (Opt.) Relative path to the trajectory file of the system (e.g. .xtc file, .trr file).
                                 If not provided, the function will only read the positions from the coordinates file.
        heavy {bool} -- (Opt.) Only extract the positions of the non-hydrogen atoms.
                        Default is True.
        begin {int} -- (Opt.) First frame to read in the trajectory. Cannot be lower than 0 or higher or equal than the final frame to read.
                       Default is 0 (first frame of the trajectory).
        end {int} -- (Opt.)Last frame to read in the trajectory. Cannot be higher or equal to the length of the trajectory or lower or equal to the first frame to read.
                     Default is the last frame of the trajectory.
        step {int} -- (Opt.)Step between frame to read. Cannot be lower or equal to 0.
                      Default is 1.
        up {bool} -- (Opt.)Check that the molecules are always orientated facing "up"
                     Default is True.
        rank {int} -- (Opt.) Number of atoms (-1) between two neighbours along the same line used for distance calculations. At rank 1, the neighbours of an atom are all atom sharing a direct bond with it.
                      Default is 6.
    Output(s):
        system {class System} -- Instance of the system classes containing all the informations on the system as well as the positions and configurations.
    """
    u = mda.Universe(topology_file, trajectory_file)
    if kwargs.get('type_info', None) is None:
        kwargs['type_info'] = pf.getTypes(u)

    # Load the positions
    positions, boxes = pf.getPos(u)

    # Initialise the class with the informations
    system_class = System(type, positions, kwargs['type_info'], boxes)

    # Get the configurations
    system_class.getCoordinates(**kwargs)
    system_class.getDistances(**kwargs)

    return system_class

# -----------------------------------------
# Generate the model for the given molecule
def generateModel(systems, phases=['ordered', 'disordered'], file_path=None, validationSize=0.20, seed=7, nSplits = 10, save_model=True):

    """Generate a model for N phases based on the input systems and save it in a file
    Argument(s):
        system {list of class System} -- List of each instances of the System class that should be used for the training. Systems can be generated using the function openSystem().
                                         Provide one system only per phase. The types, number of molecules and neighbor ranking should be the same in each system.
                                         By default, the first system of the list should be the gel one, and the second one the fluid one.
        phases {list of str} -- (Opt.) Names of the phases of each system submitted above. The order should be the same.
                                Default is gel/fluid.
        file_path {str} -- (Opt.) Path and name of the file to generate. File extension should be .lpm
                           By default, the name is autogenerated as "date_moltype.lpm" (e.g. 20201201_DPPC.lpm)
        validationSize {float} -- (Opt.) Proportion of molecules from the systems kept aside for the validation set.
                                  Default is 0.20 (20%).
        seed {int} -- (Opt.) Seed for random shuffle of the input systems.
                      Default is 7.
        nSplits {int} -- (Opt.) Number of time the training should be repeated on the training set with random shuffling.
                         Default is 10.
        save_model {bool} -- (Opt.) Save the model in a file.
                             Default is True.
    Output(s):
        models {dict of models} -- Scikit-learn models trained on the input systems for molecule classification.
    """

    # ---------------------
    # CHECK THE USER INPUTS


    # Check that the number of system matches the number of phases
    if len(systems) != len(phases):
        raise IndexError("The number of systems ("+str(len(systems))+") does not match the number of phases ("+str(len(phases))+")")

    # ----------------
    # RUN THE FUNCTION

    # Format the systems for running in the machine learning algorithms
    coordinates, distances, phases_array = _format_training_set(systems, phases=phases)

    # Train the models
    models, training_scores, training_errors = trainModel(coordinates, distances, phases_array, validationSize=validationSize, seed=seed, nSplits = nSplits)

    # Save the model
    if save_model:

        # Gather data for metadata file
        general_info = {
        'type' : systems[0].type,
        'phases' : phases
        }

        system_info = {
        'n_molecules': str(systems[0].infos['n_molecules']),
        'n_atoms_per_molecule': str(systems[0].infos['heavy_atoms']['number']),
        'n_distances_per_molecule': str(distances.shape[1]),
        'rank': str(systems[0].rank),
        }

        training_info = {
        'validation_size' : str(validationSize),
        'seed' : str(seed),
        'n_splits' : str(nSplits),
        }

        # Generate the model file
        generateModelFile(coordinates, distances, phases_array, general_info, system_info, training_info, training_scores, training_errors, file_path=file_path)

    # Merge the scores in the models
    models['scores'] = training_scores
    models['errors'] = training_errors

    return models

# ---------------------------------------------------------------
# Use a model to predict the phase of the molecules in the system
def getPhases(system, models):

    """Predict the phases of the molecules in a system based on the ML models trained previously
    Argument(s):
        system {class System} -- Instance of the system classes containing all the informations on the system as well as the positions and configurations.
        models {str or dict of models} -- Path to the model file to load or dictionary of the Scikit-Learn models to use to predict the states of the molecules.
    Output(s):
        phases {np.ndarray} -- Array of all the molecule phases predicted in the system. Dimension(s) are in (n_frames, n_molecules).
    """

    # Check the input
    if not _is_system(system):
        _error_input_type("System","instance of the System class")

    # Predict the phase of the molecules
    phases = system.getPhases(models)

    return phases

# -------------------------------------
# Set the states of the system manually
def setPhases(system, phases):

    """Set manually the phases of the molecules in a system
    Argument(s):
        system {class System} -- Instance of the system classes containing all the informations on the system as well as the positions and configurations.
        phases {str or np.ndarray} -- Phases to assign manually to the molecules.
    Output(s):
        phases {np.ndarray} -- Array of all the molecule phases predicted in the system. Dimension(s) are in (n_frames, n_molecules).
    """

    # Check the input
    if not is_system(system):
        error_input_type("System","instance of the System class")

    # Predict the phase of the molecules
    assigned_phases = system.setPhases(phases)

    return assigned_phases

# --------------------------------------------
# Save system(s) into a .csv, .xml or .h5 file
def saveSystems(systems, file_path=None, format='.csv'):

    """Save one or more system(s) in a file
    Argument(s):
        systems {class System or list of class System} -- Instances of the system classes containing the molecules to save in a file.
        file_path {str} -- (Opt.) Path and name of the file to generate. File extension should be .xml, .h5 or .csv
                           By default, the name is autogenerated as "date_hour.csv" (e.g. 20201201_012345.csv)
        format {str} -- (Opt.) File extension and format to use for the output file. Should be ".xml", ".h5" or ".csv"
                        Default is .csv
    """

    # Extract the information from the system(s)
    representation = _system_to_tessellation(systems)

    # Save the system in file
    saveRepresentation(representation, file_path=file_path, format=format)

# -------------------------------------------
# Process the system to perform tessellations
def doVoro(systems, geometry='bilayer', threshold=0.01, exclude_ghosts=None, read_neighbors=True):

    """Compute the tessellations of the system for neighbour analysis.
    Argument(s):
        systems {list of class System} -- Instances of the System classes containing the molecules to save in a file.
        geometry {str} -- (Opt.) Geometry of the system to perform the tessellations on. The current geometries available are:
                            *) bilayer - Analyse the 2D tesselations of a lipid bilayer on each leaflets.
                            *) bilayer_3d - Analyse the 3D tessellations of a lipid bilayer. Requires ghosts to have been generated first.
                            *) vesicle - Analyse the "2D" tessellations of a lipid vesicle by only keeping neighbours within the leaflet.
                                         Requires ghosts to have been generated first.
                            *) vesicle_3d - Analyse the 3D tessellations of a lipid vesicle. Requires ghosts to have been generated first.
                            *) solution - Analyse the 3D tessellations of a solution of molecules.
                          By default, the geometry is set to a (2D) bilayer.
        threshold {float} -- (Opt.) Relative area/volume threshold at which neighbours starts to be considered. Value is given as a percentage of the total area/volume.
                             Default is 0.01 (1%).
        exclude_ghosts {list of int} -- (Opt.) List of systems indices, provided with the same order than in the argument systems, that should be excluded from ghost generation.
                                        Default is None.
        read_neighbors (bool) -- (Opt.) Automatically map the local environment during the tessellation.
                                 Default is True
    Output(s):
        representation {class Tessellation} -- Instance of the class Tessellation including the representation on the system and its Voronoi tessellation.
    """

    # Convert single system in list
    if _is_system(systems):
        systems = [ systems ]

    # Check and convert input
    if not _is_list_of(systems, type='system', check_array=True, recursive=False):
        _error_input_type('Systems', 'List of System (or single System)')

    if not _is_boolean(read_neighbors):
        _error_input_type('Read neighbors', "Boolean")

    # Extract the information from the system(s)
    representation = _system_to_tessellation(systems)

    # Assign the leaflets and generate the ghosts if needed
    if geometry != "solution":

        # Get the leaflets
        representation.getLeaflets(geometry=geometry)

        # Generate the ghosts
        representation.ghosts = summonGhosts(systems, geometry=geometry, exclude_ghosts=exclude_ghosts)

    # Make the tessellation to find the neighbors
    representation.doVoronoi(geometry=geometry, threshold=threshold)

    # Read the local environment if needed
    if read_neighbors:
        representation.checkNeighbors()

    return representation

# -------------------------------------------------------------
# Save the tessellations and all related informations in a file
def saveVoro(representation, file_path=None, format='.csv'):

    """Save a representation in a file
    Argument(s):
        representation {class Tessellation} -- Instance of the class Tessellation including the representation on the system and its Voronoi tessellation.
        file_path {str} -- (Opt.) Path and name of the file to generate. File extension should be .xml, .h5 or .csv
                           By default, the name is autogenerated as "date_hour.csv" (e.g. 20201201_012345.csv)
        format {str} -- (Opt.) File extension and format to use for the output file. Should be ".xml", ".h5" or ".csv"
                        Default is .csv
    """

    # Check that the input is a Tessellation
    if not _is_tessellation(representation):
        _error_input_type('Tessellation', "instance of Tessellation class")

    # Save the system in file
    saveRepresentation(representation, file_path=file_path, format=format)





def assignLeaflets(systems, geometry='bilayer'):

    """Compute the tessellations of the system for neighbour analysis.
    Argument(s):
        systems {list of class System} -- Instances of the System classes containing the molecules to save in a file.
        geometry {str} -- (Opt.) Geometry of the system to perform the tessellations on. The current geometries available are:
                            *) bilayer - Analyse the 2D tesselations of a lipid bilayer on each leaflets.
                            *) bilayer_3d - Analyse the 3D tessellations of a lipid bilayer. Requires ghosts to have been generated first.
                            *) vesicle - Analyse the "2D" tessellations of a lipid vesicle by only keeping neighbours within the leaflet.
                                         Requires ghosts to have been generated first.
                            *) vesicle_3d - Analyse the 3D tessellations of a lipid vesicle. Requires ghosts to have been generated first.
                          By default, the geometry is set to a (2D) bilayer.
    Output(s):
        leaflets {np.ndarray} -- Array of the leaflets assigned to the membrane molecules.
    """

    # Convert single system in list
    if _is_system(systems):
        systems = [ systems ]

    # Check and convert input
    if not _is_list_of(systems, type='system', check_array=True, recursive=False):
        _error_input_type('Systems', 'List of System (or single System)')

    # Extract the information from the system(s)
    representation = _system_to_tessellation(systems)

    # Get the leaflets
    representation.getLeaflets(geometry=geometry)

    return representation.leaflets


def saveRepresentation(representation, file_path=None, format='.csv'):

    # Get the file name
    file_path = _generate_file_type(file_path, format=format, extensions=[".h5", ".xml", ".csv"])

    # Create the file
    file_name, file_extension = os.path.splitext(file_path)

    # Save in a .CSV file
    if file_extension == ".csv":
        representation_content = representation.toTable()
        representation_content.to_csv(file_path)

    # Save in a structured file (XML)
    elif file_extension == ".xml":
        representation_content = _generate_representation_content(representation)
        with open( file_path, "w" ) as xml_file:
            xml_file.write(representation_content)

    # Save in a structured file (HTF5)
    else:
        _generate_representation_binary(representation, file_path)