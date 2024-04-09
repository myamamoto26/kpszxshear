import numpy as np
import h5py
import fitsio as fio
# import swiftsimio as sw
from tqdm import tqdm
import glob
import os,sys
import time

import numba
from mpi4py import MPI
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def calculate_velocity_field(a, ne, vx, vy, vz):

    """Calculate the velocity field (* make sure it's in the right units) M=0 component.
    """

    thomson_cross = 6.9842656e-74 # Mpc^2
    tau_dot = thomson_cross * a * ne # Mpc^-1
    v2 = tau_dot * 1/np.sqrt(6) * (-vx**2 - vy**2 + 2*vz**2)

    return v2


def assign_files(nr_files, nr_ranks):
    # Taken from VirgoDC
    files_on_rank = np.zeros(nr_ranks, dtype=int)
    files_on_rank[:] = nr_files // nr_ranks
    remainder = nr_files % nr_ranks
    if remainder > 0:
        step = max(nr_files // (remainder+1), 1)
        for i in range(remainder):
            files_on_rank[(i*step) % nr_ranks] += 1
    assert sum(files_on_rank) == nr_files
    return files_on_rank


# Write values to field
@numba.jit(nopython=True, parallel=False)
def fill_field(field, indices, weight, value):
    for i in range(len(weight)):
        field[indices[i,0], indices[i,1], indices[i,2]] += value[i]

    return field

def accumulate_density_field(density_field, ptype, positions, masses, grid_size, box_size):

    """
    positions: (Npart, 3)
    masses: (Npart,)
    density_field: (grid_size, grid_size, grid_size)
    """
    scaled_positions = (positions[ptype] / box_size) * grid_size
        
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

        density_field = fill_field(density_field, affected_indices, weight, weight*masses[ptype])

    return density_field


def calculate_overdensity(density_field, apodization=False, sigma=0):
    """Calculate the overdensity field."""
    
    if apodization:
        from scipy.ndimage import gaussian_filter
        density_field = gaussian_filter(density_field, sigma=sigma)
    
    mean_density = np.mean(density_field)
    overdensity_field = (density_field / mean_density) - 1
    return overdensity_field


def accumulate_velocity_field(vx_field, vy_field, vz_field, ptype, positions, velocities, grid_size, box_size):

    """
    positions: (Npart, 3)
    masses: (Npart,)
    density_field: (grid_size, grid_size, grid_size)
    """
    scaled_positions = (positions[ptype] / box_size) * grid_size
        
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

        vx_field = fill_field(vx_field, affected_indices, weight, weight*velocities[ptype][:,0])
        vy_field = fill_field(vy_field, affected_indices, weight, weight*velocities[ptype][:,1])
        vz_field = fill_field(vz_field, affected_indices, weight, weight*velocities[ptype][:,2])

    return vx_field, vy_field, vz_field


def accumulate_any_field(field, ptype, positions, quant, grid_size, box_size):

    """
    positions: (Npart, 3)
    masses: (Npart,)
    density_field: (grid_size, grid_size, grid_size)
    """
    scaled_positions = (positions[ptype] / box_size) * grid_size
        
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

        field = fill_field(field, affected_indices, weight, weight*quant[ptype])

    return field


def main(argv):
    # Compute 3D power spectrum of matter overdensity x velocity. 
    outpath = "/cosma8/data/do012/dc-yama3/L1000N1800"
    box_size = 1000.0 # Define the size of your simulation box in the same units as positions
    box_num = 64
    grid_size = 512 # Define grid resolution
    snapshots = [i for i in range(78)] # [str(i).zfill(4) for i in range(78)]
    snap = snapshots[int(sys.argv[1])]
    # filenames = glob.glob('/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_%s/flamingo_%s.*.hdf5' % (snapshot, snapshot))

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Script parameters
    box_size = 1000
    n_part = 1800
    run = 'HYDRO_FIDUCIAL'
    part_types = [0, 1, 6] # 0: Gas, 1: DM, 4: Stars, 5: BH, 6: Neutrino

    # Determine files to read on each rank
    sim = f'L{box_size:04d}N{n_part:04d}/{run}'
    snapshot_dir = f'/cosma8/data/dp004/flamingo/Runs/{sim}/snapshots/flamingo_{snap:04}'
    if rank == 0:
        with h5py.File(f'{snapshot_dir}/flamingo_{snap:04}.0.hdf5') as file:
            n_chunk = file['Header'].attrs['NumFilesPerSnapshot'][0]
            a = file['Cosmology'].attrs['Scale-factor'][0]
        if size > n_chunk:
            print('Running with more ranks than chunk files, some ranks will be idle')
    else:
        n_chunk = 0
        a = 0
    n_chunk = comm.bcast(n_chunk)
    a = comm.bcast(a)
    files_on_rank = assign_files(n_chunk, size)
    first_file = np.cumsum(files_on_rank) - files_on_rank

    # Read files
    load_start = time.time()
    positions = {part_type: [] for part_type in part_types}
    masses = {part_type: [] for part_type in part_types}
    velocities = {0: []}
    ne = {0: []}
    for file_nr in range(
        first_file[rank], first_file[rank] + files_on_rank[rank]
        ):
        filename = f'{snapshot_dir}/flamingo_{snap:04}.{file_nr}.hdf5'
        with h5py.File(filename, 'r') as file:
            for part_type in part_types:
                positions[part_type].append(file[f'PartType{part_type}/Coordinates'][()])
                masses[part_type].append(file[f'PartType{part_type}/Masses'][()])
                if part_type == 0:
                    velocities[part_type].append(file[f'PartType{part_type}/Velocities'][()])
                    ne[part_type].append(file[f'PartType{part_type}/ElectronNumberDensities'][()])
    print(f'Load time on rank {rank:03}: {time.time()-load_start}')

    density_fields = {}
    for part_type in part_types:
        compute_start = time.time()
        # Combine data from different files on this rank
        positions[part_type] = np.concatenate(positions[part_type], axis=0)
        masses[part_type] = np.concatenate(masses[part_type], axis=0)

        # Calculate density field
        density_field = np.zeros((grid_size, grid_size, grid_size))
        density_field = accumulate_density_field(density_field, part_type, positions, masses, grid_size, box_size)
        density_fields[part_type] = density_field

        # Compute velocity field.
        if part_type == 0:
            velocities[part_type] = np.concatenate(velocities[part_type], axis=0)
            vx_field = np.zeros((grid_size, grid_size, grid_size)); vy_field = np.zeros((grid_size, grid_size, grid_size)); vz_field = np.zeros((grid_size, grid_size, grid_size))
            vx_field, vy_field, vz_field = accumulate_velocity_field(vx_field, vy_field, vz_field, part_type, positions, velocities, grid_size, box_size)
            vx = vx_field/density_field; vy = vy_field/density_field; vz = vz_field/density_field

            ne[part_type] = np.concatenate(ne[part_type], axis=0)
            ne_field = np.zeros((grid_size, grid_size, grid_size))
            ne_field = accumulate_any_field(ne_field, part_type, positions, ne, grid_size, box_size)
            v2_field = calculate_velocity_field(a, ne_field/density_field, vx, vy, vz)
            v2_field = np.nan_to_num(v2_field, nan=0.0)

        print(f'Compute time for PartType{part_type} on rank {rank:03}: {time.time()-compute_start}')
    comm.Barrier()

    all_density_field = np.zeros((grid_size, grid_size, grid_size))
    for k in density_fields.keys():
        all_density_field += density_fields[k]
    overdensity_field = calculate_overdensity(all_density_field, apodization=True, sigma=2.0)

    comm.Barrier()

    # save info on each rank.
    if not os.path.exists(os.path.join(outpath, 'snapshot_%s/overdensity_3d_snapshot_%s_%s.fits' % (snap, snap, rank))):
            fits = fio.FITS(os.path.join(outpath, 'snapshot_%s/overdensity_3d_snapshot_%s_%s.fits' % (snap, snap, rank)), 'rw')
            fits.write(overdensity_field)
            fits.close()
    print('Done writing out for overdensity')
    if not os.path.exists(os.path.join(outpath, 'snapshot_%s/velocity_3d_snapshot_%s_%s.fits' % (snap, snap, rank))):
        fits = fio.FITS(os.path.join(outpath, 'snapshot_%s/velocity_3d_snapshot_%s_%s.fits' % (snap, snap, rank)), 'rw')
        fits.write(v2_field)
        fits.close()
    print('Done writing out for velocity')


    # # send overdensity info
    # if rank != 0:
    #     print('sending overdensity info from: ', rank)
    #     comm.bcast(overdensity_field, root=0)
    # comm.Barrier()
    # if rank == 0:
    #     print('receiving overdensity info')
    #     overdensity_3d = np.zeros((box_num, grid_size, grid_size, grid_size))
    #     for i in range(1,size):
    #         tmp_res = comm.recv(source=i)
    #         overdensity_3d[i,:,:,:] = tmp_res
    # comm.Barrier()

    # # send velocity info
    # if rank != 0:
    #     print('sending velocity info from: ', rank)
    #     comm.bcast(v2_field, root=0)
    # comm.Barrier()
    # if rank == 0:
    #     print('receiving velocity info')
    #     velocity_3d = np.zeros((box_num, grid_size, grid_size, grid_size))
    #     for i in range(1,size):
    #         tmp_res = comm.recv(source=i)
    #         velocity_3d[i,:,:,:] = tmp_res
    # comm.Barrier()

    # if rank == 0:
    #     if not os.path.exists(os.path.join(outpath, 'snapshot_%s/overdensity_3d_snapshot_%s.fits' % (snapshot, snapshot))):
    #         fits = fio.FITS(os.path.join(outpath, 'snapshot_%s/overdensity_3d_snapshot_%s.fits' % (snapshot, snapshot)), 'rw')
    #         fits.write(overdensity_3d)
    #         fits.close()
    #     print('Done writing out for overdensity')
    #     if not os.path.exists(os.path.join(outpath, 'snapshot_%s/velocity_3d_snapshot_%s.fits' % (snapshot, snapshot))):
    #         fits = fio.FITS(os.path.join(outpath, 'snapshot_%s/velocity_3d_snapshot_%s.fits' % (snapshot, snapshot)), 'rw')
    #         fits.write(velocity_3d)
    #         fits.close()
    #     print('Done writing out for velocity')


if __name__ == "__main__":
    main(sys.argv)