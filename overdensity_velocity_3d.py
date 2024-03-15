import numpy as np
import h5py as h5
import fitsio as fio
import swiftsimio as sw
from tqdm import tqdm
import glob
import os

from numba import prange
from numba import njit
from numba import vectorize,guvectorize,float64,int64
# from numba.np.ufunc import parallel

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


def accumulate_density_field(positions, masses, grid_size, box_size, density_field):

    """
    positions: (Npart, 3)
    masses: (Npart,)
    density_field: (grid_size, grid_size, grid_size)
    """
    scaled_positions = (positions / box_size) * grid_size
        
    # Calculate the indices of the "lower left" corner grid point for each particle
    indices = np.floor(scaled_positions).astype(np.int32)
    
    # Calculate the distance of each particle from the "lower left" grid point in grid units
    delta = scaled_positions - indices
    
    # For each particle, distribute its mass to the surrounding 8 grid points
    for offset in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:

        # Calculate the weight for the current offset
        offset = np.array(offset)
        # weight = (1 - np.abs(delta - offset)).prod(axis=1)
        w = 1 - np.abs(delta - offset)
        weight = w[:,0] * w[:,1] * w[:,2]
        
        # Calculate the affected grid points, wrapping around using modulo for periodic boundary conditions
        affected_indices = (indices + offset) % grid_size
        
        # Use np.add.at for unbuffered in-place addition, distributing weights to the density_field
        np.add.at(density_field, 
                (affected_indices[:, 0], affected_indices[:, 1], affected_indices[:, 2]), 
                weight * masses)

    return density_field


@guvectorize([(float64[:,:], float64[:], int64, float64, int64[:], float64[:,:,:])], '(n,m),(n),(),(),(p)->(p,p,p)', nopython=True)
def accumulate_density_field_vectorized(positions, masses, grid_size, box_size, dum, density_field):

    """
    positions: (Npart, 3)
    masses: (Npart,)
    density_field: (grid_size, grid_size, grid_size)
    """
    scaled_positions = (positions / box_size) * grid_size
        
    # Calculate the indices of the "lower left" corner grid point for each particle
    indices = np.floor(scaled_positions).astype(np.int64)
    
    # Calculate the distance of each particle from the "lower left" grid point in grid units
    delta = scaled_positions - indices
    
    # For each particle, distribute its mass to the surrounding 8 grid points
    for offset in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:

        # Calculate the weight for the current offset
        offset = np.array(offset)
        # weight = (1 - np.abs(delta - offset)).prod(axis=1)
        w = 1 - np.abs(delta - offset)
        weight = w[:,0] * w[:,1] * w[:,2]
        
        # Calculate the affected grid points, wrapping around using modulo for periodic boundary conditions
        affected_indices = (indices + offset) % grid_size
        
        # Use np.add.at for unbuffered in-place addition, distributing weights to the density_field
        # np.add.at(density_field, 
        #         (affected_indices[:, 0], affected_indices[:, 1], affected_indices[:, 2]), 
        #         weight * masses)
        for i in prange(len(weight)):
            density_field[affected_indices[i,0], affected_indices[i,1], affected_indices[i,2]] += weight[i]*masses[i]

    # return density_field


def prep_density_field(filename, grid_size, box_size):
    """Create a density field using the Cloud-In-Cell (CIC) method."""

    data = sw.load(filename)

    particles = data.metadata.present_particle_names
    fields = {}
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
        elif parttype == 'stars':
            positions = data.stars.coordinates.value
            masses = data.stars.masses.value
        elif parttype == 'black_holes':
            # positions = data.black_holes.coordinates.value
            # masses = data.black_holes.masses.value
            continue
        else:
            print('unknown particle types')
            raise ValueError('Unknow particle types')

        density_field = np.zeros((grid_size, grid_size, grid_size))
        dum = np.ones(grid_size, dtype=np.int64)
        # density_field = accumulate_density_field(positions, masses, grid_size, box_size, density_field)
        density_field = accumulate_density_field_vectorized(positions, masses, grid_size, box_size, dum)

        fields[parttype] = density_field
    
    return fields


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


@guvectorize([(float64[:,:], float64[:,:], int64, float64, int64[:], float64[:,:,:], float64[:,:,:], float64[:,:,:])], '(n,m),(n,m),(),(),(p)->(p,p,p),(p,p,p),(p,p,p)', nopython=True)
def create_velocity_field_vectorized(positions, vels, grid_size, box_size, dum, vx_field, vy_field, vz_field):
    
    scaled_positions = (positions / box_size) * grid_size
    
    # Calculate the indices of the "lower left" corner grid point for each particle
    indices = np.floor(scaled_positions).astype(np.int64)
    
    # Calculate the distance of each particle from the "lower left" grid point in grid units
    delta = scaled_positions - indices
    
    # For each particle, distribute its mass to the surrounding 8 grid points
    for offset in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                   (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
        # Calculate the weight for the current offset
        offset = np.array(offset)
        w = 1 - np.abs(delta - offset)
        weight = w[:,0] * w[:,1] * w[:,2]
        
        # Calculate the affected grid points, wrapping around using modulo for periodic boundary conditions
        affected_indices = (indices + offset) % grid_size

        for i in prange(len(weight)):
            vx_field[affected_indices[i,0], affected_indices[i,1], affected_indices[i,2]] += weight[i]*vels[i,0]
            vy_field[affected_indices[i,0], affected_indices[i,1], affected_indices[i,2]] += weight[i]*vels[i,1]
            vz_field[affected_indices[i,0], affected_indices[i,1], affected_indices[i,2]] += weight[i]*vels[i,2]
    
    # return vx_field, vy_field, vz_field


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


@guvectorize([(float64[:,:], float64[:], int64, float64, int64[:], float64[:,:,:])], '(n,m),(n),(),(),(p)->(p,p,p)', nopython=True)
def create_any_field_vectorized(positions, values, grid_size, box_size, dum, field):
    
    scaled_positions = (positions / box_size) * grid_size
    
    # Calculate the indices of the "lower left" corner grid point for each particle
    indices = np.floor(scaled_positions).astype(np.int64)
    
    # Calculate the distance of each particle from the "lower left" grid point in grid units
    delta = scaled_positions - indices
    
    # For each particle, distribute its mass to the surrounding 8 grid points
    for offset in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                   (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
        # Calculate the weight for the current offset
        offset = np.array(offset)
        w = 1 - np.abs(delta - offset)
        weight = w[:,0] * w[:,1] * w[:,2]
        
        # Calculate the affected grid points, wrapping around using modulo for periodic boundary conditions
        affected_indices = (indices + offset) % grid_size

        for i in prange(len(weight)):
            field[affected_indices[i,0], affected_indices[i,1], affected_indices[i,2]] += weight[i]*values[i]

    # return field

def calculate_velocity_field(a, ne, vx, vy, vz):

    """Calculate the velocity field (* make sure it's in the right units) M=0 component.
    """

    thomson_cross = 6.9842656e-74 # Mpc^2
    tau_dot = thomson_cross * a * ne # Mpc^-1
    v2 = tau_dot * 1/np.sqrt(6) * (-vx**2 - vy**2 + 2*vz**2)

    return v2


# Compute 3D power spectrum of matter overdensity x velocity. 
outpath = "/cosma8/data/do012/dc-yama3/L1000N1800"
box_size = 1000.0 # Define the size of your simulation box in the same units as positions
box_num = 64
grid_size = 512 # Define grid resolution
snapshots = [str(i).zfill(4) for i in range(78)]
snapshot = snapshots[0]
filenames = glob.glob('/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_%s/flamingo_%s.*.hdf5' % (snapshot, snapshot))
overdensity_3d = np.zeros((box_num, grid_size, grid_size, grid_size))
velocity_3d = np.zeros((box_num, grid_size, grid_size, grid_size))
# print('processing %s files' % str(len(filenames)))
for i,filename in tqdm(enumerate(filenames)):
    all_density_field = np.zeros((grid_size, grid_size, grid_size))
    density_fields = prep_density_field(filename, grid_size, box_size) # TO-DO: implement weight interpolation. 
    for k in density_fields.keys():
        all_density_field += density_fields[k]
    overdensity_field = calculate_overdensity(all_density_field, apodization=True, sigma=2.0)
    overdensity_3d[i,:,:,:] = overdensity_field
    
    # Compute gas density weighted velocity field
    gas_positions, gas_velocities = read_particle_positions(filename, 'gas')
    gas_density_field = density_fields['gas']
    # vx, vy, vz = create_velocity_field(gas_positions, gas_velocities, grid_size, box_size)
    dum = np.ones(grid_size, dtype=np.int64)
    vx_field = np.zeros((grid_size, grid_size, grid_size)); vy_field = np.zeros((grid_size, grid_size, grid_size)); vz_field = np.zeros((grid_size, grid_size, grid_size))
    # vx_field, vy_field, vz_field = create_velocity_field(gas_positions, gas_velocities, grid_size, box_size)
    vx_field, vy_field, vz_field = create_velocity_field_vectorized(gas_positions, gas_velocities, grid_size, box_size, dum)
    vx = vx_field/gas_density_field; vy = vy_field/gas_density_field; vz = vz_field/gas_density_field
    
    a, ne = read_gas_info(filename)
    ne_field = np.zeros((grid_size, grid_size, grid_size))
    # ne_field = create_any_field(gas_positions, ne, grid_size, box_size)
    ne_field = create_any_field_vectorized(gas_positions, ne, grid_size, box_size, dum)
    v2_field = calculate_velocity_field(a, ne_field/gas_density_field, vx, vy, vz)
    v2_field = np.nan_to_num(v2_field, nan=0.0)
    velocity_3d[i,:,:,:] = v2_field

fits = fio.FITS('/cosma8/data/do012/dc-yama3/L1000N1800/overdensity_3d_snapshot_vectorized_fulltest.fits', 'rw')
fits.write(overdensity_3d)
fits.close()

fits = fio.FITS('/cosma8/data/do012/dc-yama3/L1000N1800/velocity_3d_snapshot_vectorized_fulltest.fits', 'rw')
fits.write(velocity_3d)
fits.close()

exit()
if not os.path.exists(os.path.join(outpath, 'snapshot_%s/overdensity_3d_snapshot_%s.fits' % (snapshot, snapshot))):
    fits = fio.FITS(os.path.join(outpath, 'snapshot_%s/overdensity_3d_snapshot_%s.fits' % (snapshot, snapshot)), 'rw')
    fits.write(overdensity_3d)
    fits.close()
if not os.path.exists(os.path.join(outpath, 'snapshot_%s/velocity_3d_snapshot_%s.fits' % (snapshot, snapshot))):
    fits = fio.FITS(os.path.join(outpath, 'snapshot_%s/velocity_3d_snapshot_%s.fits' % (snapshot, snapshot)), 'rw')
    fits.write(velocity_3d)
    fits.close()