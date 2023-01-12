# Import MDAnalysis and other necessary packages
import MDAnalysis as md

# Load the trajectory file and the corresponding topology file
u = md.Universe('topology.pdb', 'trajectory.xtc')

u.dimensions
# Select all atoms in the protein
protein = u.select_atoms('PROTEIN')
membrane = u.select_atoms('MEMBRANE')
solute = u.select_atoms('SOLUTE')

# Extract the positions of the atoms at every frame in the trajectory
protpos = protein.positions
mempos = membrane.positions
solpos = solute.positions
