import numpy as np
import h5py as h5
import fitsio as fio
import swiftsimio as sw
from tqdm import tqdm
import glob
import os


def compute_power_spectrum(overdensity_field, velocity_field, box_size):
    # Fourier transform the overdensity field
    delta_k = np.fft.fftn(overdensity_field)
    delta_k = np.fft.fftshift(delta_k)  # Shift zero frequency component to the center

    # Fourier transform the velocity field
    v_k = np.fft.fftn(velocity_field)
    v_k = np.fft.fftshift(v_k)  # Shift zero frequency component to the center
    
    # Compute the magnitude squared of the Fourier components
    power_spectrum = delta_k * np.conj(v_k) #np.abs(delta_k)**2
    
    # Correct for the volume of the box and the number of grid points
    norm = (box_size / overdensity_field.shape[0])**3
    power_spectrum *= norm
    
    # Compute the wave numbers corresponding to each Fourier mode
    kx = np.fft.fftfreq(overdensity_field.shape[0], d=box_size/overdensity_field.shape[0])
    ky = np.fft.fftfreq(overdensity_field.shape[1], d=box_size/overdensity_field.shape[1])
    kz = np.fft.fftfreq(overdensity_field.shape[2], d=box_size/overdensity_field.shape[2])
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    kz = np.fft.fftshift(kz)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Bin the power spectrum
    k_bins = np.logspace(np.log10(np.min(k[k>0])), np.log10(np.max(k)), num=50)
    k_val = 0.5 * (k_bins[1:] + k_bins[:-1])
    Pk = np.histogram(k.flatten(), bins=k_bins, weights=power_spectrum.flatten())[0]
    Nk = np.histogram(k.flatten(), bins=k_bins)[0]
    Pk = Pk / Nk  # Average the power spectrum in each bin
    
    return k_val, Pk


# Compute 3D power spectrum of matter overdensity x velocity. 
outpath = "/cosma8/data/do012/dc-yama3/"
box_size = 1000.0 # Define the size of your simulation box in the same units as positions
grid_size = 512 # Define grid resolution
snapshots = [str(i).zfill(4) for i in range(64)]
Pk = np.zeros((len(snapshots), 50))
print('processing %s files' % str(len(filenames)))
for i,snapshot in tqdm(enumerate(snapshots)):
    
    overdensity_3d = fio.read(os.path.join(outpath, 'overdensity_3d_snapshot_%s' % snapshot))
    velocity_3d = fio.read(os.path.join(outpath, 'velocity_3d_snapshot_%s' % snapshot))
    # Compute power spectrum for each snapshot.
    k_, Pk_ = compute_power_spectrum(overdensity_3d, velocity_3d, box_size)
    Pk[i, :] = Pk_

# Do Limber integral. 
# Compute 2D power spectrum by integrating 3D power spectrum over snapshots. 
# k, Pk = compute_power_spectrum(overdensity_field, box_size)