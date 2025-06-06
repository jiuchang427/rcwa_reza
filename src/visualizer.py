# Standard library imports
import os
import math
from typing import Tuple, List, Dict, Optional
import sys
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf
import imageio
import cv2
from PIL import Image

# Local imports
import rcwa_utils
import tensor_utils
from tensor_utils import EigGeneral
import src.solver as solver
from src.config import Config
from src.material_utils import calculate_epsilon

class Visualizer:
    """Class to handle all visualization tasks."""
    def __init__(self, config: Config):
        self.config = config
        self.loss_history = []
        self.setup_plotting()
        
    def setup_plotting(self):
        """Setup matplotlib parameters."""
        matplotlib.use('TkAgg')  # Use the TkAgg backend
        plt.rcParams.update({'font.size': 16})
        plt.rcParams['font.family'] = 'serif'

    def plot_material(self, wavelengths, n_values, k_values, eps_real, eps_imag, save_as):
        """Plot material properties."""
        # Plot n and k values
        plt.figure(figsize=(8, 6))
        plt.plot(wavelengths, n_values, 'ro', label='n (data points)', linewidth=2)
        plt.plot(wavelengths, k_values, 'go', label='k (data points)', linewidth=2)

        # Interpolate n and k values
        interp_wavelengths = np.linspace(min(wavelengths), max(wavelengths), 1000)
        interp_n = np.interp(interp_wavelengths, wavelengths, n_values)
        interp_k = np.interp(interp_wavelengths, wavelengths, k_values)
        plt.plot(interp_wavelengths, interp_n, 'b-', label='n (interpolation)', linewidth=3)
        plt.plot(interp_wavelengths, interp_k, 'c-', label='k (interpolation)', linewidth=3)

        plt.xlabel('λ (nm)')
        plt.ylabel('n, k')
        plt.legend()
        plt.title('Optical Constants')
        plt.xlim(min(wavelengths), max(wavelengths))
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, save_as + '_nk.pdf'))
        plt.show()

        # Calculate and plot epsilon
        epsilon_real = np.real(calculate_epsilon(interp_n, interp_k))
        epsilon_imag = np.imag(calculate_epsilon(interp_n, interp_k))
        plt.figure(figsize=(8, 6))
        plt.plot(interp_wavelengths, epsilon_real, 'b', label='Re(ε)', linewidth=4)
        plt.plot(interp_wavelengths, epsilon_imag, 'r', label='Im(ε)', linewidth=4)
        plt.xlabel('λ (nm)')
        plt.ylabel('ε')
        plt.legend()
        plt.title('Complex Permittivity')
        plt.xlim(min(wavelengths), max(wavelengths))
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, save_as + '_epsilon.pdf'))
        plt.show()
        return epsilon_real, epsilon_imag

    def plot_effective_permittivity(self, epsilon_am, epsilon_Cr):
        """Plot effective permittivity for different crystallization percentages."""
        Lc_values = np.linspace(0, 1, 100)
        epsilon_real_eff = []
        epsilon_imag_eff = []

        for Lc in Lc_values:
            epsilon_eff = self.calculate_effective_permittivity(Lc, epsilon_Cr, epsilon_am)
            epsilon_real_eff.append(np.real(epsilon_eff))
            epsilon_imag_eff.append(np.imag(epsilon_eff))

        plt.figure(figsize=(12, 8))
        plt.plot(Lc_values * 100, epsilon_real_eff, 'b-', label='Re(εeff)', linewidth=4)
        plt.plot(Lc_values * 100, epsilon_imag_eff, 'c-', label='Im(εeff)', linewidth=4)
        plt.xlabel('Crystallization %', fontsize=16)
        plt.ylabel('εeff', fontsize=16)
        plt.legend(fontsize=14)
        plt.title('Effective Permittivity', fontsize=18)
        plt.tight_layout()
        plt.xlim(min(Lc_values* 100), max(Lc_values* 100))
        plt.savefig(os.path.join(self.config.output_dir, 'plot_effective_permittivity.pdf'))
        plt.show()

    def plot_polarizations(self, wavelength_mat, TM_amp, TM_phase, TE_amp, TE_phase, file_name):
        """Plot polarization data."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        # Plot TM polarization
        axes[0].plot(wavelength_mat * 1e-3, TM_amp, 'k-', label='TM Relative Transmission', linewidth=4)
        axes[0].set_xlabel(r"$\lambda_{\mu m}$", fontsize=16)
        axes[0].set_ylabel('Transmission', fontsize=16)
        axes[0].set_title('Transmission and Phase vs Wavelength for TM Polarization', fontsize=18)
        axes[0].set_yticks([0, 1])
        axes[0].legend(loc='upper right', fontsize=14)
        axes[0].set_xlim(wavelength_mat[0]/1000, wavelength_mat[-1]/1000)

        ax2 = axes[0].twinx()
        ax2.plot(wavelength_mat * 1e-3, TM_phase, 'r-', label='TM Relative Phase Delay', linewidth=4)
        ax2.set_ylabel('TM Phase Delay', color='r', fontsize=16)
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax2.set_yticklabels(['$-π$', '$-π/2$', '0', '$π/2$', '$π$'], fontsize=14)
        ax2.set_xlim(wavelength_mat[0]/1000, wavelength_mat[-1]/1000)

        # Plot TE polarization
        axes[1].plot(wavelength_mat * 1e-3, TE_amp, 'g-', label='TE Relative Transmission', linewidth=4)
        axes[1].set_xlabel(r'$\lambda_{\mu m}$', fontsize=16)
        axes[1].set_ylabel('Transmission', fontsize=16)
        axes[1].set_title('Transmission and Phase vs Wavelength for TE Polarization', fontsize=18)
        axes[1].set_yticks([0, 1])
        axes[1].legend(loc='upper right', fontsize=14)
        axes[1].set_xlim(wavelength_mat[0]/1000, wavelength_mat[-1]/1000)

        ax3 = axes[1].twinx()
        ax3.plot(wavelength_mat * 1e-3, TE_phase, 'y-', label='TE Relative Phase Delay', linewidth=4)
        ax3.set_ylabel('TE Phase Delay', color='y', fontsize=16)
        ax3.tick_params(axis='y', labelcolor='y')
        ax3.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax3.set_yticklabels(['$-π$', '$-π/2$', '0', '$π/2$', '$π$'], fontsize=14)
        ax3.set_xlim(wavelength_mat[0]/1000, wavelength_mat[-1]/1000)

        plt.savefig(os.path.join(self.config.output_dir, file_name + '.pdf'), bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    def plot_real_part_effective_permittivity(self, Lx, ER_t, output_file):
        """Plot the real part of effective permittivity."""
        index = 0  # Specify the index of the Lc value you want to visualize
        real_part = ER_t[0, 0, 0, 0, :, :].real.cpu().detach().numpy()
        imag_part = ER_t[0, 0, 0, 1, :, :].real.cpu().detach().numpy()
        n = 0.5 * np.sqrt(np.sqrt(imag_part**2 + real_part**2) + real_part)
        
        # Get the dimensions of the real_part array
        height, width = real_part.shape

        # Calculate the coordinate range
        x_range = np.linspace(-width/2, width/2, width)
        y_range = np.linspace(-height/2, height/2, height)
        x_range = np.multiply(x_range, Lx/width) * 1e9
        y_range = np.multiply(y_range, Lx/height) * 1e9

        cmap = 'inferno'
        plt.figure(figsize=(10, 6))
        plt.imshow(n, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], cmap=cmap, origin='lower')
        plt.xlabel('x (nm)', fontsize=18)
        plt.ylabel('y (nm)', fontsize=18)
        plt.title('Refractive Index Profile', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.savefig(os.path.join(self.config.output_dir, output_file))
        plt.show()

    def plot_real_part_effective_permittivity_and_FOM_online(self, loss_history, iteration_history, config, ER_t, output_file, outputs):
        """Plot real part of effective permittivity and FOM during optimization."""
        Lx = config.Lx
        Ly = config.Ly
        index = 0  # Specify the index of the Lc value you want to visualize
        real_part = ER_t[0, 0, 0, 0, :, :].real.cpu().detach().numpy()
        imag_part = ER_t[0, 0, 0, 1, :, :].real.cpu().detach().numpy()
        n = 0.5 * np.sqrt(np.sqrt(imag_part**2 + real_part**2) + real_part)
        
        # Get the dimensions of the real_part array
        height, width = real_part.shape

        # Calculate the coordinate range
        x_range = np.linspace(-width/2, width/2, width)
        y_range = np.linspace(-height/2, height/2, height)
        x_range = np.multiply(x_range, Lx/width) * 1e9
        y_range = np.multiply(y_range, Ly/height) * 1e9
        x_range_order = int((outputs["T"][0,0,0,0].shape)[0]/2)
        y_range_order = int((outputs["T"][0,0,0,0].shape)[0]/2)
        x_range_order_x = np.linspace(-x_range_order, x_range_order,(outputs["T"][0,0,0,0].shape)[0] )

        # Plot data on axs0 (FOM vs Iteration)
        config.axs0.clear()  # Clear previous plot
        config.axs0.plot(iteration_history, loss_history, color='red', linewidth=3)  # Plot full history
        config.axs0.set_xlabel('Iteration', fontsize=14)
        config.axs0.set_ylabel('FOM', fontsize=14)
        config.axs0.set_title('FOM vs Iteration', fontsize=16)
        config.axs0.grid(True)  # Add grid for better visibility

        # Plot data on axs2 (Refractive Index Profile)
        cmap = 'inferno'
        im = config.axs2.imshow(n, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], cmap=cmap, origin='lower')
        config.axs2.set_xlabel('x (nm)', fontsize=14)
        config.axs2.set_ylabel('y (nm)', fontsize=14)
        config.axs2.set_title('Refractive Index Profile', fontsize=16)
        
        config.axs1.clear() 
        config.axs1.plot(x_range_order_x,(outputs["T"][0,0,0,x_range_order,:].cpu()).detach(),color='blue', linewidth=3.5, label='Transmission')
        config.axs1.plot(x_range_order_x,(outputs["R"][0,0,0,x_range_order,:].cpu()).detach(),color='green', linewidth=3.5,  label='Reflection')

        # Plotting red dots
        config.axs1.scatter(x_range_order_x,(outputs["T"][0,0,0,x_range_order,:].cpu()).detach(), color='red')
        config.axs1.scatter(x_range_order_x,(outputs["R"][0,0,0,x_range_order,:].cpu()).detach(), color='black')
        config.axs1.legend()

        # Adjust layout and save
        images_opt_dir = os.path.join(config.output_dir, 'images_opt')
        os.makedirs(images_opt_dir, exist_ok=True)
        output_path = os.path.join(images_opt_dir, output_file)
        plt.savefig(output_path)
        plt.pause(0.01)  # Pause to allow the plot to be updated
        return im

    def create_animation(self, duration: float = 0.5):
        """Create an animation from a sequence of images."""
        image_directory = os.path.join(self.config.output_dir, 'images_opt')
        output_path = os.path.join(self.config.output_dir, 'output_gif.gif')
        
        image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        
        images = []
        for image_file in image_files:
            image_path = os.path.join(image_directory, image_file)
            image = Image.open(image_path)
            images.append(image)
        
        imageio.mimsave(output_path, images, duration=duration)
        
    def create_video(self, fps: int = 10):
        """Create a video from a sequence of images."""
        image_directory = os.path.join(self.config.output_dir, 'images_opt')
        output_path = os.path.join(self.config.output_dir, 'output_video.mp4')

        image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        
        first_image = cv2.imread(os.path.join(image_directory, image_files[0]))
        height, width, layers = first_image.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for image_file in image_files:
            image_path = os.path.join(image_directory, image_file)
            image = cv2.imread(image_path)
            out.write(image)
            
        out.release()

    def plotter(self, Reflection, wavelength_mat, ylabel, output_filename, color='blue'):
        """Create a line plot with specified parameters."""
        # Create a line plot with the specified color
        plt.plot(wavelength_mat, Reflection, color=color, linewidth=3)
        # Set labels and title
        plt.xlabel('λ (nm)')
        plt.ylabel(ylabel)
        # Set plot limits
        plt.xlim(min(wavelength_mat), max(wavelength_mat))
        # Adjust figure labels within the plot
        plt.tight_layout()
        # Save the plot as a PDF file
        plt.savefig(os.path.join(self.config.output_dir, output_filename))
        # Display the plot
        plt.show()

    def plotter_iteration(self, Reflection, wavelength_mat, ylabel, output_filename, color='blue'):
        """Create a line plot for iteration data."""
        # Create a line plot with the specified color
        plt.plot(wavelength_mat, Reflection, color=color, linewidth=3)
        # Set labels and title
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        # Set plot limits
        plt.xlim(min(wavelength_mat), max(wavelength_mat))
        # Adjust figure labels within the plot
        plt.tight_layout()
        # Save the plot as a PDF file
        plt.savefig(os.path.join(self.config.output_dir, output_filename))
        # Display the plot
        plt.show()

    def plot_combined(self, Lx, ER_t, Reflection, wavelength_mat, output_file):
        """Create a combined plot of refractive index profile and reflection."""
        # Get the dimensions of the real_part array
        height, width = ER_t.shape[-2:]

        # Calculate the coordinate range
        x_range = np.linspace(-width/2, width/2, width)
        y_range = np.linspace(-height/2, height/2, height)
        x_range = np.multiply(x_range, Lx/width)*1e9
        y_range = np.multiply(y_range, Lx/height)*1e9

        # Calculate the magnitude of ER_t
        magnitude = np.sqrt(np.abs(ER_t))

        # Create the figure and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot the refractive index profile
        cmap = 'inferno'
        im1 = ax1.imshow(magnitude[0,0,0,0], extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], cmap=cmap, origin='lower')
        ax1.set_xlabel('x (nm)', fontsize=18)
        ax1.set_ylabel('y (nm)', fontsize=18)
        ax1.set_title('Refractive Index Profile', fontsize=18)
        ax1.tick_params(labelsize=16)
        fig.colorbar(im1, ax=ax1)

        # Plot the Reflection vs. Wavelength
        ax2.plot(wavelength_mat, Reflection, color='blue', linewidth=3)
        ax2.set_xlabel('λ (nm)')
        ax2.set_ylabel('Reflection')
        ax2.set_xlim(min(wavelength_mat), max(wavelength_mat))
        ax2.tick_params(labelsize=14)

        # Adjust figure labels within the plot
        fig.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.config.output_dir, output_file))

        # Display the plot
        plt.show()
    
    def material_plotter(self, filename, start_wavelength, end_wavelength, resolution):
        wavelengths, n_values, k_values = self.read_material_file(filename)
        epsilon_real,epsilon_imag = self.plot_material(wavelengths, n_values, k_values, None, None, filename.split('.')[0])
        return epsilon_real,epsilon_imag


    # %%
    def plot_epsilon_vs_wavelength_range(self, wavelengths, Lc_values):
        """
        Plot the real and imaginary parts of permittivity (ε) for different Lc values as a function of wavelength in a specified range.

        Args:
        - wavelengths: Array of wavelengths.
        - n_values: Array of refractive indices.
        - k_values: Array of extinction coefficients.
        - Lc_values: Array of Lc values.

        Returns:
        None (plots the results).
        """   
        # Increase font size
        plt.rcParams.update({'font.size': 16})
        # Set serif font
        plt.rcParams['font.family'] = 'serif'
        Lc_values = np.linspace(0, 1, 11)
        # Specify the wavelength range
        wavelength_range = np.linspace(0.4, 1.6, 400)
        # Initialize arrays to store epsilon values
        epsilon_real = np.zeros((len(Lc_values),len(wavelength_range)))
        epsilon_imag = np.zeros((len(Lc_values),len(wavelength_range)))
        for i, Lc in enumerate(Lc_values):
            for j, wavelength in enumerate(wavelength_range):
                wavelengths, n_values_am, k_values_am = self.read_material_file(r'material/AM_SB2_S3.txt')
                n_am, k_am, epsilon_am = self.interpolate_material(wavelengths, n_values_am, k_values_am, wavelength)
                wavelengths, n_values, k_values = self.read_material_file(r'material/CR_SB2_S3.txt')  
                n_cr, k_cr, epsilon_cr = self.interpolate_material(wavelengths, n_values_cr, k_values_cr, wavelength)
                epsilon_eff = self.calculate_effective_permittivity(Lc, epsilon_cr, epsilon_am)
                epsilon_real[i, j] = np.real(epsilon_eff)
                epsilon_imag[i, j] = np.imag(epsilon_eff)
        # Plot epsilon vs wavelength for different Lc values - Real part
        fig, ax = plt.subplots(figsize=(12,8))
        cmap = bipolar(neutral=0.2, interp='linear', lutsize=2048)

        colors =cmap(np.linspace(0, 1, len(Lc_values)))

        for i, Lc in enumerate(Lc_values):
            ax.plot(wavelength_range, epsilon_real[i], color=colors[i], linestyle='-', label=f'Re(ε) - Lc={Lc:.2f}')

        ax.set_xlabel('λ (nm)', fontsize=16)
        ax.set_ylabel('Permittivity (ε)', fontsize=16)
        ax.legend(fontsize=12)
        ax.set_title('Real Part of Effective Permittivity vs Wavelength', fontsize=18)
        ax.grid(True)
        plt.tight_layout()
        plt.xlim(min(wavelength_range), max(wavelength_range))

        # Create a colorbar
        norm = plt.Normalize(0, 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Lc', fontsize=14)

        # Save the plot
        plt.savefig('material/materialepsilon_real_vs_wavelength.pdf')

        # Show the plot
        plt.show()

        # Plot epsilon vs wavelength for different Lc values - Imaginary part
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = cmap(np.linspace(0, 1, len(Lc_values)))

        for i, Lc in enumerate(Lc_values):
            ax.plot(wavelength_range, epsilon_imag[i], color=colors[i], linestyle='-', label=f'Im(ε) - Lc={Lc:.2f}')

        ax.set_xlabel('λ (nm)', fontsize=16)
        ax.set_ylabel('Permittivity (ε)', fontsize=16)
        ax.legend(fontsize=12)
        ax.set_title('Imaginary Part of Effective Permittivity vs Wavelength', fontsize=18)
        ax.grid(True)
        plt.tight_layout()
        plt.xlim(min(wavelength_range), max(wavelength_range))

        # Create a colorbar
        norm = plt.Normalize(0, 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Lc', fontsize=14)

        # Save the plot
        plt.savefig('material/epsilon_imag_vs_wavelength.pdf')

        # Show the plot
        plt.show()
