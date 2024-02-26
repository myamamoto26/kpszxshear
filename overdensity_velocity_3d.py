import numpy as np
import h5py as h5
import fitsio as fio
import swiftsimio as sw
from tqdm import tqdm
import glob
import os


def read_particle_positions(file_path, particles):
    """Read particle positions from an HDF5 file."""
    # with h5.File(file_path, 'r') as f:
    #     positions = f[particles+'/Coordinates'][:] 

    data = sw.load(file_path)
    if particles == 'dm':
        positions = data.dark_matter.coordinates.value
        velocities = data.dark_matter.velocities.value
    elif particles == 'gas':
        positions = data.gas.coordinates.value
        velocities = data.gas.velocities.value

    return positions, velocities


def read_gas_info(file_path):

    data = sw.load(file_path)

    return data.metadata.scale_factor, data.gas.electron_number_densities.value

def accumulate_density_field(density_field, positions, masses, grid_size, box_size):

    scaled_positions = (positions / box_size) * grid_size
        
    # Calculate the indices of the "lower left" corner grid point for each particle
    indices = np.floor(scaled_positions).astype(int)
    
    # Calculate the distance of each particle from the "lower left" grid point in grid units
    delta = scaled_positions - indices
    
    # For each particle, distribute its mass to the surrounding 8 grid points
    for offset in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
        # Calculate the weight for the current offset
        offset = np.array(offset)
        weight = (1 - np.abs(delta - offset)).prod(axis=1)
        
        # Calculate the affected grid points, wrapping around using modulo for periodic boundary conditions
        affected_indices = (indices + offset) % grid_size
        
        # Use np.add.at for unbuffered in-place addition, distributing weights to the density_field
        np.add.at(density_field, 
                (affected_indices[:, 0], affected_indices[:, 1], affected_indices[:, 2]), 
                weight * masses)
    
    return density_field

def create_density_field(filename, grid_size, box_size, gas_only=False):
    """Create a density field using the Cloud-In-Cell (CIC) method."""
    
    density_field = np.zeros((grid_size, grid_size, grid_size))

    data = sw.load(filename)
    if not gas_only:
        particles = data.metadata.present_particle_names
    else:
        particles = ['gas']

    for parttype in particles:
        print('particle type, ', parttype)
        if parttype == 'dark_matter':
            positions = data.dark_matter.coordinates.value
            masses = data.dark_matter.masses.value
        elif parttype == 'gas':
            positions = data.gas.coordinates.value
            masses = data.gas.masses.value
        elif parttype == 'neutrinos':
            positions = data.neutrinos.coordinates.value
            masses = data.neutrinos.masses.value
        else:
            print('unknown particle types')
            raise ValueError('Unknow particle types')

        density_field = accumulate_density_field(density_field, positions, masses, grid_size, box_size)
    
    return density_field


def calculate_overdensity(density_field, apodization=False, sigma=0):
    """Calculate the overdensity field."""
    
    if apodization:
        from scipy.ndimage import gaussian_filter
        density_field = gaussian_filter(density_field, sigma=sigma)
    
    mean_density = np.mean(density_field)
    overdensity_field = (density_field / mean_density) - 1
    return overdensity_field

def create_velocity_field(positions, vels, grid_size, box_size):

    """Calculate average velocity in a cell"""
    vx_field = np.zeros((grid_size, grid_size, grid_size))
    vy_field = np.zeros((grid_size, grid_size, grid_size))
    vz_field = np.zeros((grid_size, grid_size, grid_size))
    
    scaled_positions = (positions / box_size) * grid_size
    
    # Calculate the indices of the "lower left" corner grid point for each particle
    indices = np.floor(scaled_positions).astype(int)
    
    # Calculate the distance of each particle from the "lower left" grid point in grid units
    delta = scaled_positions - indices
    
    # For each particle, distribute its mass to the surrounding 8 grid points
    for offset in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                   (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
        # Calculate the weight for the current offset
        offset = np.array(offset)
        weight = (1 - np.abs(delta - offset)).prod(axis=1)
        
        # Calculate the affected grid points, wrapping around using modulo for periodic boundary conditions
        affected_indices = (indices + offset) % grid_size
        
        # Use np.add.at for unbuffered in-place addition, distributing weights to the density_field
        np.add.at(vx_field, 
                  (affected_indices[:, 0], affected_indices[:, 1], affected_indices[:, 2]), 
                  weight * vels[:,0])
        np.add.at(vy_field, 
                  (affected_indices[:, 0], affected_indices[:, 1], affected_indices[:, 2]), 
                  weight * vels[:,1])
        np.add.at(vz_field, 
                  (affected_indices[:, 0], affected_indices[:, 1], affected_indices[:, 2]), 
                  weight * vels[:,2])
    
    return vx_field, vy_field, vz_field


def create_any_field(positions, values, grid_size, box_size):

    field = np.zeros((grid_size, grid_size, grid_size))
    
    scaled_positions = (positions / box_size) * grid_size
    
    # Calculate the indices of the "lower left" corner grid point for each particle
    indices = np.floor(scaled_positions).astype(int)
    
    # Calculate the distance of each particle from the "lower left" grid point in grid units
    delta = scaled_positions - indices
    
    # For each particle, distribute its mass to the surrounding 8 grid points
    for offset in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                   (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
        # Calculate the weight for the current offset
        offset = np.array(offset)
        weight = (1 - np.abs(delta - offset)).prod(axis=1)
        
        # Calculate the affected grid points, wrapping around using modulo for periodic boundary conditions
        affected_indices = (indices + offset) % grid_size
        
        # Use np.add.at for unbuffered in-place addition, distributing weights to the density_field
        np.add.at(field, 
                  (affected_indices[:, 0], affected_indices[:, 1], affected_indices[:, 2]), 
                  weight * values)

    return field

def calculate_velocity_field(a, ne, vx, vy, vz):

    """Calculate the velocity field (* make sure it's in the right units) M=0 component.
    """

    thomson_cross = 6.9842656e-74 # Mpc^2
    tau_dot = thomson_cross * a * ne # Mpc^-1
    v2 = tau_dot * 1/np.sqrt(6) * (-vx**2 - vy**2 + 2*vz**2)

    return v2


# Compute 3D power spectrum of matter overdensity x velocity. 
outpath = "/cosma8/data/do012/dc-yama3/"
box_size = 1000.0 # Define the size of your simulation box in the same units as positions
box_num = 64
grid_size = 512 # Define grid resolution
snapshots = [str(i).zfill(4) for i in range(64)]
snapshot = snapshots[0] # Just look at snapshot-0000 for now. Will parallelize this for running this over all the snapshots. 
filenames = glob.glob('/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_%s/flamingo_%s.*.hdf5' % (snapshot, snapshot))
overdensity_3d = np.zeros((box_num, grid_size, grid_size, grid_size))
velocity_3d = np.zeros((box_num, grid_size, grid_size, grid_size))
print('processing %s files' % str(len(filenames)))
for i,filename in tqdm(enumerate(filenames)):

    density_field = create_density_field(filename, grid_size, box_size) # TO-DO: implement weight interpolation. 
    overdensity_field = calculate_overdensity(density_field, apodization=True, sigma=2.0)
    overdensity_3d[i,:,:,:] = overdensity_field

    # Compute gas density weighted velocity field
    gas_positions, gas_velocities = read_particle_positions(filename, 'gas')
    gas_density_field = create_density_field(filename, grid_size, box_size, gas_only=True)
    vx, vy, vz = create_velocity_field(gas_positions, gas_velocities, grid_size, box_size)
    vx = vx/gas_density_field; vy = vy/gas_density_field; vz = vz/gas_density_field

    a, ne = read_gas_info(filename)
    ne_field = create_any_field(gas_positions, ne, grid_size, box_size)
    v2_field = calculate_velocity_field(a, ne_field/gas_density_field, vx, vy, vz)
    velocity_3d[i,:,:,:] = v2_field

if not os.path.exists(os.path.join(outpath, 'overdensity_3d_snapshot_%s.fits' % snapshot)):
    fits = fio.FITS(os.path.join(outpath, 'overdensity_3d_snapshot_%s.fits' % snapshot), 'rw')
    fits.write(overdensity_3d)
    fits.close()
if not os.path.exists(os.path.join(outpath, 'velocity_3d_snapshot_%s.fits' % snapshot)):
    fits = fio.FITS(os.path.join(outpath, 'velocity_3d_snapshot_%s.fits' % snapshot), 'rw')
    fits.write(velocity_3d)
    fits.close()