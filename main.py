"""Main script for RCWA simulation."""

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
from src.optimizer import Optimizer
from src.visualizer import Visualizer
import src.solver as solver

def main():
    """Main function to run the optimization process."""
    print("Starting optimization...")
    config = Config()
    
    optimizer = Optimizer(config)
    optimizer.initialize()
    
    optimizer.optimize()
    
    visualizer = Visualizer(config)
    visualizer.plot_real_part_effective_permittivity_and_FOM_online(
        optimizer.loss_history,
        optimizer.iteration_history,
        config,
        optimizer.effective_permittivity,
        "final_result.png",
        optimizer.simulation_outputs
    )
    
    # Create animations
    visualizer.create_animation(duration=0.5)
    visualizer.create_video(fps=10)
    
    # Plot final results
    TM_amp, TE_amp, TM_phase, TE_phase, wavelength_mat, ER_t = optimizer.Spectrum_phase_amp(
        optimizer.final_pattern,
        UR_t,
        delta=1*(5)+1,
        start_wavelength=1530,
        end_wavelength=1570
    )
    
    visualizer.plot_polarizations(
        wavelength_mat,
        TM_amp,
        TM_phase,
        TE_amp,
        TE_phase,
        'final_results'
    )

    # Additional binary analysis
    print("\nPerforming binary structure analysis...")
    threshold = 8  # Set the threshold value
    eps_min = config.eps_min   # Minimum value for thresholding
    eps_max = config.eps_max  # Maximum value for thresholding
    
    # Create binary structure using optimizer's method
    binary_ER_t = optimizer.apply_threshold(optimizer.effective_permittivity, threshold, eps_min, eps_max)
    
    # Visualize binary structure
    visualizer.plot_real_part_effective_permittivity(
        config.Lx,
        binary_ER_t,
        'binary_structure.png'
    )
    
    # Analyze binary structure performance
    print("Analyzing binary structure performance...")
    TM_amp_binary, TE_amp_binary, TM_phase_binary, TE_phase_binary, wavelength_mat_binary, _ = optimizer.Spectrum_phase_amp(
        binary_ER_t,
        UR_t,
        delta=1*(5)+1,
        start_wavelength=1530,
        end_wavelength=1570
    )

if __name__ == "__main__":
    main() 