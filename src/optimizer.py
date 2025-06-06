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
from src.config import Config
import src.solver as solver
from src.visualizer import Visualizer
from src.material_utils import calculate_epsilon

class Optimizer:
    """Class to handle the optimization process."""
    def __init__(self, config: Config):
        self.config = config
        self.pattern = None
        self.eps_r = None
        self.eps_r_ref = None
        self.optimizer = None
        self.loss_history = []
        self.iteration_history = []
        self.visualizer = Visualizer(config)
        self.device = None
        
        # Optimization results
        self.final_pattern = None
        self.effective_permittivity = None
        self.effective_permeability = None
        self.simulation_outputs = None
        
    def configure_cuda(self):
        """Configure CUDA settings and initialize device."""
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        torch.set_printoptions(precision=8)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        torch.cuda.empty_cache()
        
    def initialize(self):
        """Initialize the optimization parameters."""
        # Configure CUDA
        self.configure_cuda()
        
        # Generate initial pattern
        self.pattern = self.random_start(
            self.config.Nx, 
            self.config.Ny, 
            (1, 1), 
            self.config.rand_params, 
            False, 
            False
        )
        self.pattern = torch.nn.Parameter(self.pattern.to(self.device))
        self.pattern.requires_grad = True
        
        # Generate epsilon tensors
        self.eps_r = self.generate_eps_r(
            self.config.Nx,
            self.config.Ny,
            self.config.eps_min,
            self.config.eps_max,
            False,
            False
        )
        self.eps_r = torch.nn.Parameter(self.eps_r.to(self.device))
        
        self.eps_r_ref = self.generate_eps_r(
            self.config.Nx,
            self.config.Ny,
            self.config.eps_min,
            self.config.eps_min,
            False,
            False
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam([self.eps_r], lr=5e-2)
        
    def optimize(self):
        """Run the optimization process."""
        b_vector = self.generate_bvector(self.config.iteration_max, self.config.bin_params)
        threshold_vectors, n_robustness = self.generate_thresh_vectors(self.config.opt_params)
        
        for iteration in range(self.config.iteration_max):
            self.optimizer.zero_grad()
            
            # Apply filters
            filtered_pattern = self.density_filter_2d(self.pattern, self.config.BlurGridLarge)
            binary_pattern = self.thresh_filter(filtered_pattern, b_vector[iteration], 0.5)
            self.pattern.data = self.replace_nan_with_1(binary_pattern)
            
            # Update epsilon
            self.eps_r = (self.pattern.squeeze(0).squeeze(0) * 
                         (self.config.eps_max - self.config.eps_min)) + self.config.eps_min
            
            # Calculate loss
            loss_val, outputs, ER_t, UR_t = self.loss_func(self.eps_r, self.config)
            loss_val.backward(retain_graph=True)
            self.optimizer.step()
            
            # Update pattern
            self.pattern.data = self.enforce_symmetry(self.pattern, False, False)
            self.pattern.data = torch.clamp(self.pattern, min=0, max=1)
            
            # Record history
            self.loss_history.append(loss_val.item())
            self.iteration_history.append(iteration)
            # Plot progress using visualizer
            self.visualizer.plot_real_part_effective_permittivity_and_FOM_online(
                self.loss_history,
                self.iteration_history,
                self.config,
                ER_t,
                f"{iteration}.png",
                outputs
            )
            
            print(f'Iteration: {iteration+1}/{self.config.iteration_max}, FOM: {loss_val.item()}')
        
        # Store final results
        self.final_pattern = self.get_final_pattern()
        self.effective_permittivity = ER_t
        self.effective_permeability = UR_t
        self.simulation_outputs = outputs
        
        return self
    
    def get_final_pattern(self):
        """Get the final optimized pattern."""
        filtered_pattern = self.density_filter_2d(self.pattern, self.config.BlurGridLarge)
        binary_pattern = self.thresh_filter(filtered_pattern, self.config.bin_params['Max'], 0.5)
        filtered_pattern2 = self.gauss_filter_2d(binary_pattern, self.config.BlurGrid)
        return self.thresh_filter(filtered_pattern, self.config.bin_params['Max'], 0.5)

    def read_material_file(self, filename):
        data = np.loadtxt(filename, delimiter=',')
        wavelengths = data[:, 0]
        n_values = data[:, 1]
        k_values = data[:, 2]
        return wavelengths, n_values, k_values

    def interpolate_material(self, wavelengths, n_values, k_values, wavelength):
        n = np.interp(wavelength, wavelengths, n_values)
        k = np.interp(wavelength, wavelengths, k_values)
        epsilon = calculate_epsilon(n, k)
        return n, k, epsilon

    # %%
    def calculate_effective_permittivity(self, Lc, epsilon_c, epsilon_a):
        """
        Calculate the effective permittivity of partially crystallized GST by solving the equation numerically using the bisection method.

        Args:
        - Lc: Crystallization fraction of GST, ranging from 0 (amorphous) to 1 (fully crystalline).
        - epsilon_c: Permittivity of crystalline GST.
        - epsilon_a: Permittivity of amorphous GST.
        - tol: Tolerance for convergence (default: 1e-2).

        Returns:
        - Effective permittivity (εeff).
        """
        right_side = (Lc-1)*(1-epsilon_a) / (2+epsilon_a)+(Lc * ((epsilon_c - 1) / (epsilon_c + 2)))
        epsilon_eff = ((2*right_side)+1)/(1-right_side)
        return epsilon_eff
    # %%
    def loss_func_spec(self, ER_t, UR_t,params,desired_phase_1,desired_phase_2,transmitted_field_ref):
        # Generate permittivity and permeability distributions.
        PQ_zero = torch.tensor(params["PQ"]).prod() // 2
        ## Simulation
        outputs = solver.simulate(ER_t, UR_t, params)
        tx = outputs["tx"][:, :, :, PQ_zero, 0] # Get the zero order field by PQ_zero
        ty = outputs["ty"][:, :, :, PQ_zero, 0] 
        field = torch.unsqueeze(torch.transpose(torch.stack((tx, ty)), 0, 1), 0)
        #   plot_real_part_effective_permittivity(params["Lx"],ER_t, 'refIndex.pdf')
        transmitted_field = torch.squeeze(field)
        transmitted_field = transmitted_field
        relative_Transmission = torch.abs((transmitted_field)/ np.abs(transmitted_field_ref))**2
        relative_Phase_delay = torch.angle(torch.exp(1j* (torch.angle(transmitted_field))-np.angle(transmitted_field_ref)))
        #   transmitted_field = transmitted_field-transmitted_field_ref
        desired_phase_1 = torch.deg2rad(desired_phase_1).to("cuda")
        desired_phase_2 =  torch.deg2rad(desired_phase_2).to("cuda")
        arg_1 = -1j*(torch.log(transmitted_field/torch.sqrt(torch.multiply(transmitted_field,torch.conj(transmitted_field)))))
        arg_1 = arg_1.to("cuda")
        dif = (torch.cos(desired_phase_1)-torch.cos(arg_1))+(torch.sin(desired_phase_1)-torch.sin(arg_1))
        #   print(arg_1-relative_Phase_delay)
        FOM =(1-relative_Transmission)*(dif)
        return FOM,relative_Transmission,relative_Phase_delay,ER_t, UR_t

    # %%
    def gaussian_kernel(self, size: int, std: float):
        """Generate a 2D Gaussian kernel."""
        coords = torch.linspace(-size, size, 2*size+1)
        g = torch.exp(-(coords**2) / (2*std**2))
        g_norm = g / g.sum()
        g2d = g_norm[:, None] * g_norm[None, :]
        return g2d
    def generate_eps_r(self, Nx, Ny, eps_min, eps_max, asymmetry_y=False, asymmetry_x=False):
        mean = abs(eps_min + eps_max) / 2  # Set the mean of the normal distribution
        std = abs(eps_max - eps_min) / 1  # Set the standard deviation of the normal distribution
        
        # Generate eps_r based on the given conditions
        if asymmetry_y and asymmetry_x:
            square_size = Nx // 2
            square_eps_r = torch.normal(mean, std, size=(square_size, square_size))
            flipped_eps_r = torch.flip(square_eps_r, dims=[0, 1])
            eps_r = torch.cat((torch.cat((flipped_eps_r, flipped_eps_r.flip(1)), dim=1),
                    torch.cat((flipped_eps_r.flip(0), flipped_eps_r.flip(0).flip(1)), dim=1)), dim=0)

        elif asymmetry_y:
            square_size = Ny // 2
            square_eps_r = torch.normal(mean, std, size=(square_size, Ny))
            eps_r = torch.cat((square_eps_r, square_eps_r.flip(0)), dim=0)

        elif asymmetry_x:
            square_size = Nx // 2
            square_eps_r = torch.normal(mean, std, size=(Nx, square_size))
            eps_r = torch.cat((square_eps_r, square_eps_r.flip(1)), dim=1)
            
        else:
            eps_r = torch.normal(mean, std, size=(Nx, Ny))

        # Applying convolution to smooth the eps_r tensor
        kernel = self.gaussian_kernel(3, 2)  # Define the Gaussian kernel
        kernel = kernel[None, None, :, :]  # Add extra dimensions to the kernel for batch and channel
        eps_r = eps_r[None, None, :, :]    # Add batch and channel dimensions to eps_r
        eps_r = F.conv2d(eps_r, kernel, padding=3)
        eps_r = eps_r.squeeze()  # Remove batch and channel dimensions

        eps_r.requires_grad = True

        return eps_r

    # %%
    def loss_func_binary_ER_t(self, binary_ER_t,UR_t,params):
        outputs = solver.simulate(binary_ER_t, UR_t, params)
        # Maximize the reflectance.
        ref_lambda1 = (outputs['REF'][0, 0, 0])
        return (1- ref_lambda1),binary_ER_t,UR_t

    # %%
    def apply_threshold(self, ER_t, threshold, eps_min, eps_max):
        # Split the complex tensor into real and imaginary parts
        ER_real = ER_t.real.to("cpu")
        ER_imag = ER_t.imag.to("cpu")

        # Apply the thresholding condition
        binary_ER_real = torch.where(ER_real < threshold, eps_min, eps_max)
        binary_ER_imag = torch.where(ER_imag < threshold, eps_min, eps_max)

        # Combine the thresholded real and imaginary parts
        binary_ER_t = binary_ER_real + 1j * binary_ER_imag

        return binary_ER_t
    # %%
    def Spectrum(self, Lx,delta,start_wavelength,end_wavelength):
        Reflection = []
        wavelength_mat = []
        epsilon_mat = []
        thetas_mat = []
        stepnum = int((end_wavelength-start_wavelength)/delta)+1
        wavelengths, n_values, k_values = self.read_material_file(r'material/HamedAbr-TiO2 (1).csv')#AM_SB2_S3.txt')
        params = solver.initialize_params(wavelengths = [1550])  
        for i in range (0,stepnum):
            wavelength = start_wavelength+i*delta
            n, k, epsilon = self.interpolate_material(wavelengths, n_values, k_values, wavelength*1e-3)
            wavelength_mat.append(wavelength)
            epsilon_mat.append(np.conj(epsilon))
            thetas_mat.append(0.0)
        for i in range (0,stepnum):
            params = solver.initialize_params(wavelengths = [wavelength_mat[i]], thetas=[0.0],
                                            erd=[epsilon_mat[i]],ers=2.25, PQ=[11, 11],
                                    L=[400, 2500.0], Lx=950.0, Ly=950.0, Nx=128)
            Transmission = self.loss_func_binary_ER_t(binary_ER_t,UR_t,params)
            Reflection.append(1-Transmission[0].item())
            if i%5 == 0:
                print("simulating wavelength:",wavelength_mat[i])

        Reflection = np.array(Reflection)
        wavelength_mat = np.array(wavelength_mat)
        return Reflection,wavelength_mat,ER_t
    # %%
    def density_filter_2d(self, pattern_in, radius):
        # If the radius is less than 1 pixel, no filter can be applied
        if radius < 1:
            pattern_out = pattern_in
        else:
            # Define grid
            x = torch.arange(-int(torch.ceil(torch.tensor(float(radius))).item()),
                            int(torch.ceil(torch.tensor(float(radius))).item()) + 1,
                            device=pattern_in.device)
            X1, X2 = torch.meshgrid(x, x)
            # Compute weights
            weights = radius - torch.sqrt(X1**2 + X2**2)
            weights[weights < 0] = 0
            b = torch.sum(weights)
            # Apply filter
            pattern_out = F.conv2d(pattern_in.unsqueeze(0).unsqueeze(0).to(pattern_in.device),
                                weights.unsqueeze(0).unsqueeze(0).to(pattern_in.device) / b,
                                padding=int(radius))
            pattern_out = pattern_out.squeeze(0).squeeze(0)
        return pattern_out

    # %%
    def adjust_eps_r(self, eps_r, threshold=0.5, increment=0.01):
        """
        Adjusts elements of a PyTorch tensor eps_r based on a threshold.

        Args:
            eps_r (torch.Tensor): Input PyTorch tensor.
            threshold (float): Threshold value.
            increment (float): Value to add or subtract from elements based on the threshold.

        Returns:
            torch.Tensor: Adjusted tensor.
        """
        # Create masks for elements greater and smaller than the threshold
        above_threshold = eps_r > threshold
        below_threshold = eps_r < threshold
        # Apply adjustments
        adjusted_eps_r = eps_r.clone()  # Create a copy to avoid modifying the original tensor
        adjusted_eps_r[above_threshold] += increment
        adjusted_eps_r[below_threshold] -= increment
        return adjusted_eps_r

    # %%
    def generate_bvector(self, max_iterations, bin_parm):
        # Extract parameters
        b_min = bin_parm['Min']# Minimum value of B that is used after it is initialized.
        b_max = bin_parm['Max']# Maximum value of B
        b_start = bin_parm['IterationStart'] #Iteration at which B is allowed to be nonzero.
        b_hold = bin_parm['IterationHold'] #Number of iterations during which B is kept constant before increasing (after it is initialized)
        b_mid = b_max / 20
        b_vector = torch.zeros(max_iterations)
        bmult1 = (b_mid / b_min) ** (1 / torch.floor(torch.tensor((round(max_iterations / 2) - b_start) / b_hold)))
        bmult2 = (b_max / b_mid) ** (1 / torch.floor(torch.tensor((round(max_iterations / 2)) / b_hold)))
        # The binarization speed is a piecewise function
        b_vector[b_start:round(max_iterations / 2)] = b_min * bmult1 ** (torch.floor((torch.arange(round(max_iterations / 2) - b_start) + 1) / b_hold))
        b_vector[round(max_iterations / 2):] = b_mid * bmult2 ** (torch.floor((torch.arange(max_iterations - round(max_iterations / 2)) + 1) / b_hold))
        return b_vector

    # %%
    def custom_gaussian_blur(self, image, kernel_size, sigma):
        """
        Custom Gaussian blur implementation using 2D convolution.
        """
        # Create a 1D Gaussian kernel
        x = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size)
        gauss_1d = torch.exp(-0.5 * (x / sigma).pow(2))
        gauss_1d /= gauss_1d.sum()
        
        # Create a 2D Gaussian kernel
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        gauss_2d = gauss_2d / gauss_2d.sum()
        
        # Expand dimensions to match the input tensor
        gauss_2d = gauss_2d[None, None, :, :]
        
        # Apply Gaussian blur
        blurred_img = F.conv2d(image[None, None, :, :], gauss_2d, padding=kernel_size // 2)[0][0]
        
        return blurred_img
    def enforce_symmetry(self, pattern, sym_x, sym_y):
        """
        Enforce symmetry on the pattern.
        """
        if sym_x:
            pattern = (pattern + pattern.flip(-2)) / 2
        if sym_y:
            pattern = (pattern + pattern.flip(-1)) / 2
        return pattern
    def random_start(self, n_x, n_y, period, rand_parm, sym_x, sym_y):
        """
        Generate a random starting pattern.
        """
        pitch = rand_parm['Pitch']
        rand_average = rand_parm['Average']
        rand_sigma = rand_parm['Sigma']
        n_cells_x = math.ceil(2 * period[0] / pitch)
        n_cells_y = math.ceil(2 * period[1] / pitch)
        grid_size = period[0] / n_x

        # Normally distributed index of refractions
        random_indices = rand_average * torch.ones(n_cells_x, n_cells_y) + rand_sigma * torch.randn(n_cells_x, n_cells_y)
        random_indices = self.enforce_symmetry(random_indices, sym_x, sym_y)

        # Upsample the random pattern
        random_pattern = F.interpolate(random_indices[None, None, :, :], size=(n_x, n_y), mode='bilinear')[0][0]

        # Gaussian blur the pattern
        blur_size = int(1.0 * pitch / grid_size)
        random_pattern = self.custom_gaussian_blur(random_pattern, blur_size, 0.3)

        # Ensure the pattern is proper
        random_pattern = self.enforce_symmetry(random_pattern, sym_x, sym_y)
        random_pattern = torch.clamp(random_pattern, 0, 1)
        random_pattern.requires_grad = True

        return random_pattern

    # %%
    def thresh_filter(self, pattern_in, bin, midpoint):
        """
        Apply a threshold filter to the input pattern.
        """
        bin = bin.detach().clone().requires_grad_(True) if isinstance(bin, torch.Tensor) else torch.tensor(bin, requires_grad=True)
        midpoint = midpoint.detach().clone().requires_grad_(True) if isinstance(midpoint, torch.Tensor) else torch.tensor(midpoint, requires_grad=True)
        
        if bin != 0:
            patt_norm_low = 1 - pattern_in / midpoint
            pattern_low = midpoint * (torch.exp(-bin * patt_norm_low) - patt_norm_low * torch.exp(-bin))
            
            patt_norm_high = (pattern_in - midpoint) / (1 - midpoint)
            pattern_high = midpoint + (1 - midpoint) * (1 - torch.exp(-bin * patt_norm_high) + patt_norm_high * torch.exp(-bin))
        else:
            pattern_low = pattern_in / (2 * midpoint)
            pattern_high = (pattern_in - 1) / (2 - 2 * midpoint) + 1
        
        pattern_out = pattern_low * (pattern_in <= midpoint) + pattern_high * (pattern_in > midpoint)

        return pattern_out

    # %%
    def generate_thresh_vectors(self, opt_parm):
        """
        Generate thresholding parameters for robustness.

        Args:
        - opt_parm (dict): Dictionary containing optimization parameters.

        Returns:
        - threshold_vectors (Tensor): Tensor containing thresholding parameters.
        - n_robustness (int): Number of robustness parameters.
        """
        max_iterations = opt_parm['Optimization']['Iterations']
        start = torch.tensor(opt_parm['Optimization']['Robustness']['StartDeviation'])
        end = torch.tensor(opt_parm['Optimization']['Robustness']['EndDeviation'])
        ramp = opt_parm['Optimization']['Robustness']['Ramp']
        n_robustness = len(start)

        if n_robustness != len(end):
            raise ValueError('Robustness vectors are not the same length!')

        deviation_vectors = torch.zeros((n_robustness, max_iterations))

        for ii in range(n_robustness):
            deviation_vectors[ii, :ramp] = torch.linspace(start[ii], end[ii], ramp)
            deviation_vectors[ii, ramp:] = end[ii]

        blur_radius = opt_parm['Optimization']['Filter']['BlurRadius']
        threshold_vectors = 0.5 * (1 - torch.from_numpy(erf(deviation_vectors.numpy() / blur_radius)))

        return threshold_vectors, n_robustness

    # %%
    def gauss_filter_2d(self, pattern_in, blure_radius):
        # Create a Gaussian kernel using PyTorch's inbuilt functions
        # Note: This creates a 1D Gaussian kernel; to get the 2D version, we'll take an outer product
        gauss_1d = torch.exp(-torch.linspace(-2*blure_radius, 2*blure_radius, 4*blure_radius+1)**2 / (2*blure_radius**2)).to("cuda")
        gauss_2d = torch.outer(gauss_1d, gauss_1d).to("cuda")
        
        # Normalize the kernel to make sure the weights sum to 1
        gauss_2d /= gauss_2d.sum()
        
        # Adjust pattern_in to have an extra dimension for convolution to work
        pattern_in = pattern_in[None, None, :, :]
        
        # Apply the Gaussian filter using PyTorch's functional API
        # The padding is done to handle boundaries (circular padding is simulated by periodic padding)
        filtered_pattern = F.conv2d(pattern_in, gauss_2d[None, None, :, :], padding=blure_radius)
        filtered_pattern = filtered_pattern.squeeze()

        # Squeeze out the extra dimensions
        return filtered_pattern

    # %%
    def pad_to_size_with_ones(self, tensor, target_size):
        """
        Pads a 2D tensor to the given target size using ones.
        """
        # Compute padding amounts
        pad_diff = torch.tensor(target_size) - torch.tensor(tensor.size())
        pad_left = pad_diff // 2
        pad_right = pad_diff - pad_left
        
        # Apply padding
        padded_tensor = F.pad(tensor, (pad_left[1].item(), pad_right[1].item(), pad_left[0].item(), pad_right[0].item()), value=1)
        
        return padded_tensor


    # %%
    def pad_to_size_with_zeros(self, tensor, target_size):
        """
        Pads a 2D tensor to the given target size using ones.
        """
        # Compute padding amounts
        pad_diff = torch.tensor(target_size) - torch.tensor(tensor.size())
        pad_left = pad_diff // 2
        pad_right = pad_diff - pad_left
        
        # Apply padding
        padded_tensor = F.pad(tensor, (pad_left[1].item(), pad_right[1].item(), pad_left[0].item(), pad_right[0].item()), value=0)
        
        return padded_tensor

    # %%
    def gauss_grad_2d(self, gradient_in, pattern_in, bin_val, midpoint, sigma):
        """
        Computes the base pattern gradient from the Gaussian and threshold filtered gradient.
        """
        bin_val = torch.tensor(bin_val)
        
        # Compute the derivative of the threshold filter
        if bin_val != 0:
            pattern_low = bin_val * (torch.exp(-bin_val * (1 - pattern_in / midpoint))) + torch.exp(-bin_val)
            pattern_high = torch.exp(-bin_val) + bin_val * torch.exp(-bin_val * (pattern_in - midpoint) / (1 - midpoint))
        else:
            pattern_low = torch.ones_like(pattern_in) / (2 * midpoint)
            pattern_high = torch.ones_like(pattern_in) / (2 * (1 - midpoint))
        
        # Combine the two pieces of the threshold derivative
        pattern_low[pattern_in > midpoint] = 0
        pattern_high[pattern_in <= midpoint] = 0
        pattern_deriv = pattern_low + pattern_high
        
        # Apply chain rule
        gradient = pattern_deriv * gradient_in
        
        # Chain rule of Gaussian filter is another Gaussian
        gradient_out = self.density_filter_2d(gradient, sigma)
        
        return gradient_out

    # %%
    def filtered_grad_2d(self, gradient_in, pattern_in, bin_val, midpoint, radius):
        """
        Compute the gradient from the density and threshold filtered gradient.
        """
        
        # Convert scalars to tensors
        bin_tensor = torch.tensor(bin_val, dtype=gradient_in.dtype, device=gradient_in.device)
        
        # Compute the derivative of the threshold filter
        if bin_val != 0:
            pattern_low = bin_tensor * (torch.exp(-bin_tensor * (1 - pattern_in / midpoint))) + torch.exp(-bin_tensor)
            pattern_high = torch.exp(-bin_tensor) + bin_tensor * torch.exp(-bin_tensor * (pattern_in - midpoint) / (1 - midpoint))
        else:
            pattern_low = torch.ones_like(pattern_in) / (2 * midpoint)
            pattern_high = torch.ones_like(pattern_in) / (2 * (1 - midpoint))
        
        # Combine the two pieces of the threshold derivative
        pattern_low[pattern_in > midpoint] = 0
        pattern_high[pattern_in <= midpoint] = 0
        pattern_deriv = pattern_low + pattern_high
        
        # Apply chain rule
        gradient = pattern_deriv * gradient_in
        
        # Chain rule of the DensityFilter is another DensityFilter
        gradient_out = self.density_filter_2d(gradient, radius)
        return gradient_out


    # %%
    def create_disk_filter(self, radius):
        # Ensure radius is an integer
        radius = int(radius)
        # Generate the grid
        y, x = np.ogrid[-radius: radius+1, -radius: radius+1]
        # Create the mask
        mask = x**2 + y**2 <= radius**2
        # Create the disk using the mask
        disk = np.zeros((2*radius+1, 2*radius+1))
        disk[mask] = 1
        # Normalize and return as a torch tensor
        return torch.tensor(disk / disk.sum(), dtype=torch.float32)

    def blur_geom_post_grad(self, device_pattern, iteration, opt_parm, grid_scale,max_iterations):    
        # Large blur every X iterations
        BlurLargeIter = 6
        BlurLargeIterstop = 3
        BlurRadiusLarge = 6
        BlurSmallIter = 3
        BlurSmallIterStop = 6
        BlurRadiusSmall = 3
        if (iteration % BlurLargeIter == 0) and (iteration < max_iterations -BlurLargeIterstop ):
            filter_large = self.create_disk_filter(0.5 * int(BlurRadiusLarge / grid_scale)).cuda()
            device_pattern = F.conv2d(device_pattern[None, None, :, :], filter_large[None, None, :, :], padding=filter_large.shape[0] // 2)[0, 0]
            device_pattern = device_pattern.to("cuda") 
        # Small blur every Y iterations
        elif (iteration % BlurSmallIter== 0) and (iteration < max_iterations - BlurSmallIterStop):
            filter_small = (self.create_disk_filter(BlurRadiusSmall / grid_scale)).to("cuda") 
            device_pattern = F.conv2d(device_pattern[None, None, :, :], filter_small[None, None, :, :], padding=filter_small.shape[0] // 2)[0, 0]
            device_pattern = device_pattern.to("cuda")   
        return device_pattern


    # %%
    def define_grid(self, grid, period, wavelength):
        """
        Compute the simulation grid for given geometry.
        """
        
        # Number of grid points
        n_grid = np.ceil(np.array(grid) * np.array(period) / wavelength).astype(int)
        nx, ny = n_grid
        
        # Device period
        px, py = period
        
        # Compute external grid coordinates
        x_bounds = np.linspace(0, px, nx+1)
        y_bounds = np.linspace(0, py, ny+1)
        
        # Compute size of each grid box
        dx = x_bounds[1] - x_bounds[0]
        dy = y_bounds[1] - y_bounds[0]
        
        # Compute coordinates of center of each box
        x_grid = x_bounds[1:] - 0.5 * dx
        y_grid = y_bounds[1:] - 0.5 * dy
        
        # Compute average grid size
        dr = (np.mean([dx, dy]))
        
        return x_grid, y_grid, dr

    # %%
    def replace_nan_with_1(self, average_final_pattern):
        """
        Replace NaN values in a PyTorch tensor with 1.

        Args:
            average_final_pattern (torch.Tensor): Input PyTorch tensor.

        Returns:
            torch.Tensor: Tensor with NaN values replaced by 1.
        """
        # Create a mask for NaN values
        nan_mask = torch.isnan(average_final_pattern)

        # Replace NaN values with 1
        average_final_pattern[nan_mask] = 0

        return average_final_pattern

    # %%
    def loss_func(self, eps_r, config):
        """Calculate the loss function for optimization.
        
        Args:
            eps_r: The permittivity tensor to optimize
            config: Configuration object containing simulation parameters
            
        Returns:
            tuple: (loss value, outputs, ER_t, UR_t, outputs)
        """
        # Generate permittivity and permeability distributions
        ER_t, UR_t = solver.generate_arbitrary_epsilon(eps_r, config)
        PQ_zero = int(config.PQ[0]/2)
        
        # Simulation
        outputs = solver.simulate(ER_t, UR_t, config)
        tx = outputs["tx"][:, :, :, PQ_zero, 0]  # Get the zero order field by PQ_zero
        ty = outputs["ty"][:, :, :, PQ_zero, 0] 
        field = torch.unsqueeze(torch.transpose(torch.stack((tx, ty)), 0, 1), 0)
        
        transmitted_field = torch.squeeze(field)
        transmitted_field = transmitted_field[0]
        FOM = outputs["T"][0,0,0,PQ_zero,PQ_zero+1]
        return 1-FOM, outputs, ER_t, UR_t

    # %%

    def Spectrum_phase_amp(self, ER_t, UR_t, delta, start_wavelength, end_wavelength, config):
        stepnum = int((end_wavelength-start_wavelength)/delta)+1
        wavelengths, n_values, k_values = self.read_material_file('material\HamedAbr-TiO2 (1).csv')
        
        # Generate all wavelengths at once
        wavelength_mat = np.array([start_wavelength + i*delta for i in range(stepnum)])
        
        # Calculate epsilon for all wavelengths
        epsilon_mat = []
        for wavelength in wavelength_mat:
            n, k, epsilon = self.interpolate_material(wavelengths, n_values, k_values, wavelength*1e-3)
            epsilon_mat.append(np.conj(epsilon))
        epsilon_mat = np.array(epsilon_mat)
        
        # Initialize parameters for all wavelengths at once
        params = solver.initialize_params(wavelengths = wavelength_mat.tolist(),   
                                    thetas = [0.0] * stepnum,
                                    phis= [0.0] * stepnum,
                                    pte= [1.0] * stepnum,  # put real small number to avoid inf
                                    ptm= [1.0] * stepnum,
                                    erd=epsilon_mat.tolist(),
                                    ers=1.69, 
                                    PQ=[7, 7],
                                    L = [300.0, 2500.0],
                                    Lx = 2000.0,
                                    Ly = 500.0,
                                    Nx = 256)
        
        # Convert desired phases to tensors
        desired_phase_1, desired_phase_2 = self._convert_desired_phases_to_tensors(config)
        
        # Process all wavelengths at once
        loss_step, relative_Transmission, relative_Phase_delay, ER_t, UR_t = self.loss_func_spec(ER_t, UR_t, params, desired_phase_1, desired_phase_2, transmitted_field_ref)
        
        # Convert results to numpy arrays
        relative_Transmission = relative_Transmission.cpu().detach().numpy()
        relative_Phase_delay = relative_Phase_delay.cpu().detach().numpy()
        
        # Extract TM and TE components
        TM_phase = relative_Phase_delay[:, 0]
        TE_phase = relative_Phase_delay[:, 1]
        TM_amp = relative_Transmission[:, 0]
        TE_amp = relative_Transmission[:, 1]
        
        print("TM_phase:", TM_phase)
        print("TE_phase:", TE_phase)
        print("TM_amp:", TM_amp)
        print("TE_amp:", TE_amp)
        
        return TM_amp, TE_amp, TM_phase, TE_phase, wavelength_mat, ER_t

    def _convert_desired_phases_to_tensors(self, config):
        """Convert desired phases from config to tensors with correct properties.
        
        Args:
            config: Configuration object containing desired phases
            
        Returns:
            tuple: (desired_phase_1_tensor, desired_phase_2_tensor)
        """
        var_shape = (1,)  # Shape for single value
        desired_phase_1 = config.desired_phase_1 * np.ones(shape=var_shape)
        desired_phase_1 = torch.tensor(desired_phase_1, dtype=torch.float32, requires_grad=True)
        
        desired_phase_2 = config.desired_phase_2 * np.ones(shape=var_shape)
        desired_phase_2 = torch.tensor(desired_phase_2, dtype=torch.float32, requires_grad=True)
        
        return desired_phase_1, desired_phase_2
