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


class Config:
    """Configuration class to manage simulation and optimization parameters."""
    def __init__(self):
        # Create timestamped output directory at initialization
        self.output_dir = self.create_output_directory()
        
        # Material parameters
        self.eps_min = 1.0
        self.eps_max = 16.0 - 0.0j
        self.erd = 17.64 - 0.0j
        self.ers = 1.69
        self.ur1 = 1.0  # permeability in reflection region
        self.er1 = 1.0  # permittivity in reflection region
        self.ur2 = 1.0  # permeability in transmission region
        self.er2 = 1.0  # permittivity in transmission region
        self.urd = 1.0  # permeability of device
        
        # Geometry parameters
        self.Lx = 2750.0
        self.Ly = 1250.0
        self.Nx = 256
        self.Ny = int(np.round(self.Nx * self.Ly / self.Lx))
        self.pixelsX = 1
        self.pixelsY = 1
        
        # Simulation parameters
        self.wavelengths = torch.tensor(np.arange(1550, 1551, 2), dtype=torch.float32, requires_grad=True)
        self.thetas = torch.tensor([0.0] * len(self.wavelengths), dtype=torch.float32, requires_grad=True)
        self.phis = torch.tensor([0.0] * len(self.wavelengths), dtype=torch.float32, requires_grad=True)
        self.pte = torch.tensor([0.0] * len(self.wavelengths), dtype=torch.complex64, requires_grad=True)
        self.ptm = torch.tensor([1.0] * len(self.wavelengths), dtype=torch.complex64, requires_grad=True)
        self.PQ = [9, 9]
        self.L = torch.tensor([320.0, 2500.0], dtype=torch.float32, requires_grad=True)
        
        # Optimization parameters
        self.iteration_max = 5
        self.bin_params = {
            'Min': 1,
            'Max': 20.0,
            'IterationStart': 1,
            'IterationHold': 3
        }
        self.opt_params = {
            'Optimization': {
                'Iterations': self.iteration_max,
                'Robustness': {
                    'StartDeviation': [-5, 0, 5],
                    'EndDeviation': [-5, 0, 5],
                    'Ramp': 2,
                    'Weights': [0.5, 1, 0.5]
                },
                'Filter': {'BlurRadius': 3}
            }
        }
        
        # Random pattern parameters
        self.rand_params = {
            'Pitch': 0.14,
            'Average': 0.5,
            'Sigma': 0.95
        }
        
        # Blur parameters
        self.BlurGridLarge = 5
        self.BlurGrid = 2
        
        # Desired phase parameters
        self.desired_phase_1 = 180
        self.desired_phase_2 = 270
        self.desired_amp_1 = 1
        self.desired_amp_2 = 1
        
        # Visualization parameters
        self.fig = plt.figure(figsize=(16, 6))
        self.pos0 = [0.06, 0.1, 0.28, 0.8]
        self.pos1 = [0.38, 0.1, 0.28, 0.8]
        self.pos2 = [0.715, 0.1, 0.28, 0.8]
        self.axs0 = self.fig.add_axes(self.pos0)
        self.axs1 = self.fig.add_axes(self.pos1)
        self.axs2 = self.fig.add_axes(self.pos2)

        # Additional parameters
        self.nanometers = 1e-9
        self.degrees = np.pi / 180
        self.batchSize = len(self.wavelengths)
        self.Nlay = len(self.L)
        self.sigmoid_coeff = 1000.0
        self.rectangle_power = 200
        self.upsample = 1
        self.duty_min = 0.05
        self.duty_max = 0.45
        self.blur_radius = 100.0 * self.nanometers
        self.length_min = 0.1
        self.length_max = 5.0
        
        # Convert parameters to tensors
        self._convert_to_tensors()
    
    def _convert_to_tensors(self):
        """Convert parameters to tensors for simulation."""
        # Convert wavelengths
        self.lam0 = self.nanometers * self.wavelengths
        self.lam0 = self.lam0[:, None, None, None, None, None]
        self.lam0 = self.lam0.repeat(1, self.pixelsX, self.pixelsY, 1, 1, 1)
        
        # Convert angles
        self.theta = self.degrees * self.thetas
        self.theta = self.theta[:, None, None, None, None, None]
        self.theta = self.theta.repeat(1, self.pixelsX, self.pixelsY, 1, 1, 1)
        
        self.phi = self.degrees * self.phis
        self.phi = self.phi[:, None, None, None, None, None]
        self.phi = self.phi.repeat(1, self.pixelsX, self.pixelsY, 1, 1, 1)
        
        # Convert polarization
        self.pte = self.pte[:, None, None, None]
        self.pte = self.pte.repeat(1, self.pixelsX, self.pixelsY, 1)
        
        self.ptm = self.ptm[:, None, None, None]
        self.ptm = self.ptm.repeat(1, self.pixelsX, self.pixelsY, 1)
        
        # Convert layer thicknesses
        self.L = self.L[None, None, None, :, None, None]
        self.L = self.L * self.nanometers
        
        # Move all tensors to CUDA
        self._move_to_cuda()
    
    def _move_to_cuda(self):
        """Move all tensors to CUDA device."""
        tensor_attrs = [
            'lam0', 'theta', 'phi', 'pte', 'ptm', 'L',
            'wavelengths', 'thetas', 'phis', 'pte', 'ptm'
        ]
        for attr in tensor_attrs:
            if hasattr(self, attr):
                tensor = getattr(self, attr)
                if isinstance(tensor, torch.Tensor):
                    setattr(self, attr, tensor.to("cuda"))

    @staticmethod
    def create_output_directory():
        """Create a timestamped output directory."""
        timestamp = datetime.datetime.now().strftime("%m%d%y_%H%M")
        output_dir = os.path.join('output', timestamp)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images_opt'), exist_ok=True)
        return output_dir


