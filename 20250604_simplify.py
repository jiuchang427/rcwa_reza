# %%
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import torch
import numpy as np
import rcwa_utils
import tensor_utils
from tensor_utils import EigGeneral
import matplotlib.font_manager as font_manager
import matplotlib.cm as cm
import solver
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.font_manager as fm
import scipy.optimize as optimize
import scipy.interpolate
from matplotlib import cm
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erf
import torch.nn.functional as F
import math

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

torch.set_printoptions(precision=8)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU available")
else:
    device = torch.device("cpu")
    print("GPU not available")
torch.cuda.empty_cache()

# %%
import numpy as np
import matplotlib.pyplot as plt

def read_material_file(filename):
    data = np.loadtxt(filename, delimiter=',')
    wavelengths = data[:, 0]
    n_values = data[:, 1]
    k_values = data[:, 2]
    return wavelengths, n_values, k_values

def calculate_epsilon(n, k):
    epsilon_r = n**2 - k**2
    epsilon_i = 2 * n * k
    epsilon = epsilon_r + 1j * epsilon_i
    return epsilon

def interpolate_material(wavelengths, n_values, k_values, wavelength):
    n = np.interp(wavelength, wavelengths, n_values)
    k = np.interp(wavelength, wavelengths, k_values)
    epsilon = calculate_epsilon(n, k)
    return n, k, epsilon

def plot_material(wavelengths, n_values, k_values, eps_real, eps_imag, save_as):
    # Increase font size
    plt.rcParams.update({'font.size': 16})
    # Set serif font
    plt.rcParams['font.family'] = 'serif'

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
    plt.savefig(save_as + '_nk.pdf')
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
    plt.savefig(save_as + '_epsilon.pdf')
    plt.show()
    return epsilon_real,epsilon_imag

def material_plotter(filename, start_wavelength, end_wavelength, resolution):
    wavelengths, n_values, k_values = read_material_file(filename)
    epsilon_real,epsilon_imag = plot_material(wavelengths, n_values, k_values, None, None, filename.split('.')[0])
    return epsilon_real,epsilon_imag

# %%
def calculate_effective_permittivity(Lc, epsilon_c, epsilon_a):
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
def plot_effective_permittivity(epsilon_am, epsilon_Cr):
    """
    Plot the effective permittivity (εeff) for different crystallization percentages.

    Args:
    - epsilon_am: Permittivity of amorphous GST.
    - epsilon_Cr: Permittivity of crystalline GST.

    Returns:
    None (plots the results and saves as a PDF).
    """

    # Increase font size
    plt.rcParams.update({'font.size': 16})
    # Set serif font
    plt.rcParams['font.family'] = 'serif'
    # Increase the plot size
    plt.figure(figsize=(12, 8))

    Lc_values = np.linspace(0, 1, 100)
    epsilon_real_eff = []
    epsilon_imag_eff = []

    for Lc in Lc_values:
        epsilon_eff = calculate_effective_permittivity(Lc, epsilon_Cr, epsilon_am)
        epsilon_real_eff.append(np.real(epsilon_eff))
        epsilon_imag_eff.append(np.imag(epsilon_eff))

    # Plot the results
    plt.plot(Lc_values * 100, epsilon_real_eff, 'b-', label='Re(εeff)', linewidth=4)
    plt.plot(Lc_values * 100, epsilon_imag_eff, 'c-', label='Im(εeff)', linewidth=4)
    plt.xlabel('Crystallization %', fontsize=16)
    plt.ylabel('εeff', fontsize=16)
    plt.legend( fontsize=14)
    plt.title('Effective Permittivity', fontsize=18)
    plt.tight_layout()
    plt.xlim(min(Lc_values* 100), max(Lc_values* 100))
    # Save the plot as PDF
    plt.savefig('material\plot_effective_permittivity.pdf')

    # Show the plot
    plt.show()

# %%
def plot_epsilon_vs_wavelength_range(wavelengths, Lc_values):
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
            wavelengths, n_values_am, k_values_am = read_material_file('material\AM_SB2_S3.txt')
            n_am, k_am, epsilon_am = interpolate_material(wavelengths, n_values_am, k_values_am, wavelength)
            wavelengths, n_values, k_values = read_material_file('material\CR_SB2_S3.txt')  
            n_cr, k_cr, epsilon_cr = interpolate_material(wavelengths, n_values_cr, k_values_cr, wavelength)
            epsilon_eff = calculate_effective_permittivity(Lc, epsilon_cr, epsilon_am)
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
    plt.savefig('material\materialepsilon_real_vs_wavelength.pdf')

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
    plt.savefig('material\epsilon_imag_vs_wavelength.pdf')

    # Show the plot
    plt.show()
    
# %%
def plotter(Reflection, wavelength_mat, ylabel, output_filename, color='blue'):
    # Increase font size
    plt.rcParams.update({'font.size': 14})
    # Set serif font
    plt.rcParams['font.family'] = 'serif'
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
    plt.savefig(output_filename)
    # Display the plot
    plt.show()

# %%
def plotter_iteration(Reflection, wavelength_mat, ylabel, output_filename, color='blue'):
    # Increase font size
    plt.rcParams.update({'font.size': 14})
    # Set serif font
    plt.rcParams['font.family'] = 'serif'
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
    plt.savefig(output_filename)
    # Display the plot
    plt.show()

# %%
def plot_combined(Lx, ER_t, Reflection, wavelength_mat, output_file):
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

    # Create the output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    output_path = os.path.join('output', output_file)
    plt.savefig(output_path)

    # Display the plot
    plt.show()

# %%
def plot_real_part_effective_permittivity(Lx,ER_t, output_file):
    index = 0  # Specify the index of the Lc value you want to visualize
    real_part = ER_t[0, 0, 0, 0, :, :].real.cpu().detach().numpy()
    imag_part = ER_t[0, 0, 0, 1, :, :].real.cpu().detach().numpy()
    n = 0.5 * np.sqrt(np.sqrt(imag_part**2 + real_part**2) + real_part)
    
    # Get the dimensions of the real_part array
    height, width = real_part.shape

    # Calculate the coordinate range
    x_range = np.linspace(-width/2, width/2, width)
    y_range = np.linspace(-height/2, height/2, height)
    x_range = np.multiply(x_range,Lx/width)*1e9
    y_range = np.multiply(y_range,Lx/height)*1e9

    # Increase font size
    plt.rcParams.update({'font.size': 16})
    # Set serif font
    plt.rcParams['font.family'] = 'serif'
    cmap = 'inferno'
    plt.figure(figsize=(10, 6))
    # Plot the refractive index profile with centered axes
    plt.imshow(n, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], cmap = cmap,origin='lower')
#     plt.colorbar()
    plt.xlabel('x (nm)', fontsize=18)
    plt.ylabel('y (nm)', fontsize=18)
    plt.title('Refractive Index Profile', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    # Create the output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    output_path = os.path.join('output', output_file)
    plt.savefig(output_path)
    plt.show()

# %%
def plot_real_part_effective_permittivity_and_FOM(FOM, iteration, Lx, ER_t, output_file,
                                                  relative_Transmission, relative_Phase_delay,
                                                 desired_phase_1,desired_phase_2,
                                                 desired_amp_1,desired_amp_2):
    index = 0  # Specify the index of the Lc value you want to visualize
    real_part = ER_t[0, 0, 0, 0, :, :].real.cpu().detach().numpy()
    imag_part = ER_t[0, 0, 0, 1, :, :].real.cpu().detach().numpy()
    n = 0.5 * np.sqrt(np.sqrt(imag_part**2 + real_part**2) + real_part)
    # Define subplot positions
#     pos0 = [0.05, 0.1, 0.28, 0.8]
#     pos1 = [0.37, 0.1, 0.28, 0.8]
#     pos2 = [0.69, 0.1, 0.28, 0.8]
    # Get the dimensions of the real_part array
    height, width = real_part.shape

    # Calculate the coordinate range
    x_range = np.linspace(-width/2, width/2, width)
    y_range = np.linspace(-height/2, height/2, height)
    x_range = np.multiply(x_range, Lx/width) * 1e9
    y_range = np.multiply(y_range, Lx/height) * 1e9
    # Create the figure
#     fig = plt.figure(figsize=(14, 6))

    # Create subplots with the defined positions
#     axs0 = fig.add_axes(pos0)
#     axs1 = fig.add_axes(pos1, projection='polar')
#     axs2 = fig.add_axes(pos2)

    # Plot data on axs0 (FOM vs Iteration)
    axs0.plot(iteration, FOM, color='red', linewidth=3)
    axs0.set_xlabel('Iteration', fontsize=16)
    axs0.set_ylabel('FOM', fontsize=16)
    axs0.set_title('FOM vs Iteration', fontsize=18)

    # Plot data on axs2 (Refractive Index Profile)
    cmap = 'inferno'
    axs2.imshow(n, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], cmap=cmap, origin='lower')
    axs2.set_xlabel('x (nm)', fontsize=16)
    axs2.set_ylabel('y (nm)', fontsize=16)
    axs2.set_title('Refractive Index Profile', fontsize=18)

    # Plot data on axs1 (Polar plot)
    area = 120
    axs1.scatter((relative_Phase_delay.cpu().item()), relative_Transmission.cpu().item(), color='red',  s=area, alpha=1, label='RCWA Crystalline Phase', marker="d")
    area = 400
    axs1.scatter(np.deg2rad(desired_phase_1), desired_amp_1, color='green', s=area, alpha=1, label='Desired Crystalline Phase', marker="d")

    # Adjust layout and save
    plt.tight_layout()
    if not os.path.exists('output'):
        os.makedirs('output')
    output_path = os.path.join('output', output_file)
    plt.savefig(output_path)
    plt.show()

# %%
def plot_real_part_effective_permittivity_and_FOM_online(FOM, iteration,Lx,Ly, ER_t, output_file,
                                                  outputs):
    index = 0  # Specify the index of the Lc value you want to visualize
    real_part = ER_t[0, 0, 0, 0, :, :].real.cpu().detach().numpy()
    imag_part = ER_t[0, 0, 0, 1, :, :].real.cpu().detach().numpy()
    n = 0.5 * np.sqrt(np.sqrt(imag_part**2 + real_part**2) + real_part)
    # Define subplot positions
#     pos0 = [0.05, 0.1, 0.28, 0.8]
#     pos1 = [0.37, 0.1, 0.28, 0.8]
#     pos2 = [0.69, 0.1, 0.28, 0.8]
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
    # Create the figure
#     fig = plt.figure(figsize=(14, 6))

    # Create subplots with the defined positions
#     axs0 = fig.add_axes(pos0)
#     axs1 = fig.add_axes(pos1, projection='polar')
#     axs2 = fig.add_axes(pos2)

    # Plot data on axs0 (FOM vs Iteration)
    axs0.plot(iteration, FOM, color='red', linewidth=3)
    axs0.set_xlabel('Iteration', fontsize=14)
    axs0.set_ylabel('FOM', fontsize=14)
    axs0.set_title('FOM vs Iteration', fontsize=16)

    # Plot data on axs2 (Refractive Index Profile)
    cmap = 'inferno'
    im =  axs2.imshow(n, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]], cmap=cmap, origin='lower')
    axs2.set_xlabel('x (nm)', fontsize=14)
    axs2.set_ylabel('y (nm)', fontsize=14)
    axs2.set_title('Refractive Index Profile', fontsize=16)
    
    # Add a colorbar to the heatmap
#     plt.colorbar(heatmap, ax=axs2)
#     axs1.imshow(outputs["T"][0,0,0].cpu().detach(), extent=[-x_range_order, x_range_order, -y_range_order, y_range_order], cmap=cmap, origin='lower')
    axs1.clear() 
    axs1.plot(x_range_order_x,(outputs["T"][0,0,0,x_range_order,:].cpu()).detach(),color='blue', linewidth=3.5, label='Transmission')
    axs1.plot(x_range_order_x,(outputs["R"][0,0,0,x_range_order,:].cpu()).detach(),color='green', linewidth=3.5,  label='Reflection')

    # Plotting red dots
    axs1.scatter(x_range_order_x,(outputs["T"][0,0,0,x_range_order,:].cpu()).detach(), color='red')  # Adjust color as needed
    axs1.scatter(x_range_order_x,(outputs["R"][0,0,0,x_range_order,:].cpu()).detach(), color='black')  # Adjust color as needed
    axs1.legend()

    # Adjust layout and save
    plt.tight_layout()
    if not os.path.exists('output\\images_opt'):
        os.makedirs('output\\images_opt')
    output_path = os.path.join('output\\images_opt', output_file)
    plt.savefig(output_path)
    plt.show()
    plt.pause(0.01)  # Pause to allow the plot to be updated
    return im

# %%
def loss_func_ref(eps_r_ref):
  global params
    # Generate permittivity and permeability distributions.
  ER_t, UR_t = solver.generate_arbitrary_epsilon(eps_r_ref, params)
  PQ_zero = torch.tensor(params["PQ"]).prod() // 2
  ## Simulation
  outputs = solver.simulate(ER_t, UR_t, params)
  tx = outputs["tx"][:, :, :, PQ_zero, 0] # Get the zero order field by PQ_zero
  ty = outputs["ty"][:, :, :, PQ_zero, 0] 
  field = torch.unsqueeze(torch.transpose(torch.stack((tx, ty)), 0, 1), 0)
#   plot_real_part_effective_permittivity(params["Lx"],ER_t, 'refIndex.pdf')
  transmitted_field = torch.squeeze(field)
  transmitted_field = transmitted_field[0]
  return transmitted_field.item()

# %%
def loss_func_spec(ER_t, UR_t,params,desired_phase_1,desired_phase_2,transmitted_field_ref):
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
def gaussian_kernel(size: int, std: float):
    """Generate a 2D Gaussian kernel."""
    coords = torch.linspace(-size, size, 2*size+1)
    g = torch.exp(-(coords**2) / (2*std**2))
    g_norm = g / g.sum()
    g2d = g_norm[:, None] * g_norm[None, :]
    return g2d
def generate_eps_r(Nx, Ny, eps_min, eps_max, asymmetry_y=False, asymmetry_x=False):
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
    kernel = gaussian_kernel(3, 2)  # Define the Gaussian kernel
    kernel = kernel[None, None, :, :]  # Add extra dimensions to the kernel for batch and channel
    eps_r = eps_r[None, None, :, :]    # Add batch and channel dimensions to eps_r
    eps_r = F.conv2d(eps_r, kernel, padding=3)
    eps_r = eps_r.squeeze()  # Remove batch and channel dimensions

    eps_r.requires_grad = True

    return eps_r
# %%
# Initialize global `params` dictionary storing optimization and simulation settings.
params = solver.initialize_params(wavelengths = [1550.0],
                                  thetas = [0.0],
                                  erd =12.64, # Negative imaginary part convention for loss
                                          ers = 1.69 ,
                                  PQ = [7, 7],
                                  L = [ 400.0, 2500.0],
                                  Lx = 1260.0,
                                  Ly = 1260.0,
                                  Nx = 128)
N_x = params['Nx']
N_y = params['Ny']
eps_min = params['eps_min']
eps_max = params['eps_max']
asymmetry_y = False
asymmetry_x = False
eps_r = generate_eps_r(N_x,N_y, eps_min, eps_max, asymmetry_y, asymmetry_x)
eps_r_ref = generate_eps_r(N_x,N_y, eps_min, eps_min, asymmetry_y, asymmetry_x)
transmitted_field_ref = loss_func_ref(eps_r_ref)

# %%
def loss_func_binary_ER_t(binary_ER_t,UR_t,params):
  outputs = solver.simulate(ER_t, UR_t, params)
  # Maximize the reflectance.
  ref_lambda1 = (outputs['REF'][0, 0, 0])
  return (1- ref_lambda1),ER_t,UR_t

# %%
def apply_threshold(ER_t, threshold, eps_min, eps_max):
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
def Spectrum(Lx,delta,start_wavelength,end_wavelength):
    Reflection = []
    wavelength_mat = []
    epsilon_mat = []
    thetas_mat = []
    stepnum = int((end_wavelength-start_wavelength)/delta)+1
    wavelengths, n_values, k_values = read_material_file('material\AM_SB2_S3.txt')
    params = solver.initialize_params(wavelengths = [1550])  
    for i in range (0,stepnum):
        wavelength = start_wavelength+i*delta
        n, k, epsilon = interpolate_material(wavelengths, n_values, k_values, wavelength*1e-3)
        wavelength_mat.append(wavelength)
        epsilon_mat.append(np.conj(epsilon))
        thetas_mat.append(0.0)
    for i in range (0,stepnum):
        params = solver.initialize_params(wavelengths = [wavelength_mat[i]], thetas=[0.0],
                                          erd=[epsilon_mat[i]],ers=2.25, PQ=[11, 11],
                                  L=[400, 2500.0], Lx=950.0, Ly=950.0, Nx=128)
        Transmission = loss_func_binary_ER_t(binary_ER_t,UR_t,params)
        Reflection.append(1-Transmission[0].item())
        if i%5 == 0:
            print("simulating wavelength:",wavelength_mat[i])
#             print("simulating epsilon:",epsilon_mat[i])

    Reflection = np.array(Reflection)
    wavelength_mat = np.array(wavelength_mat)
    return Reflection,wavelength_mat,ER_t
# %%
def density_filter_2d(pattern_in, radius):
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
def adjust_eps_r(eps_r, threshold=0.5, increment=0.01):
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
def generate_bvector(max_iterations, bin_parm):
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
def custom_gaussian_blur(image, kernel_size, sigma):
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
def enforce_symmetry(pattern, sym_x, sym_y):
    """
    Enforce symmetry on the pattern.
    """
    if sym_x:
        pattern = (pattern + pattern.flip(-2)) / 2
    if sym_y:
        pattern = (pattern + pattern.flip(-1)) / 2
    return pattern
def random_start(n_x, n_y, period, rand_parm, sym_x, sym_y):
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
    random_indices = enforce_symmetry(random_indices, sym_x, sym_y)

    # Upsample the random pattern
    random_pattern = F.interpolate(random_indices[None, None, :, :], size=(n_x, n_y), mode='bilinear')[0][0]

    # Gaussian blur the pattern
    blur_size = int(1.0 * pitch / grid_size)
    random_pattern = custom_gaussian_blur(random_pattern, blur_size, 0.3)

    # Ensure the pattern is proper
    random_pattern = enforce_symmetry(random_pattern, sym_x, sym_y)
    random_pattern = torch.clamp(random_pattern, 0, 1)
    random_pattern.requires_grad = True

    return random_pattern

# %%
def thresh_filter(pattern_in, bin, midpoint):
    """
    Apply a threshold filter to the input pattern.
    """
    bin = torch.tensor(bin)
    midpoint = torch.tensor(midpoint)
    
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
def generate_thresh_vectors(opt_parm):
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
def gauss_filter_2d(pattern_in, blure_radius):
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
def pad_to_size_with_ones(tensor, target_size):
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
def pad_to_size_with_zeros(tensor, target_size):
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
def gauss_grad_2d(gradient_in, pattern_in, bin_val, midpoint, sigma):
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
    gradient_out = density_filter_2d(gradient, sigma)
    
    return gradient_out

# %%
def filtered_grad_2d(gradient_in, pattern_in, bin_val, midpoint, radius):
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
    gradient_out = density_filter_2d(gradient, radius)
    return gradient_out

# %%
def enforce_symmetry(pattern_in, sym_x, sym_y):
    """
    Enforces required symmetries in the pattern by folding over midline and averaging.
    """
    pattern_out = pattern_in.clone()
    
    # Enforce X symmetry
    if sym_x:
        pattern_out = 0.5 * (pattern_out + torch.flip(pattern_out, [0]))
    
    # Enforce Y symmetry
    if sym_y:
        pattern_out = 0.5 * (pattern_out + torch.flip(pattern_out, [1]))
        
    return pattern_out


# %%
def create_disk_filter(radius):
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

def blur_geom_post_grad(device_pattern, iteration, opt_parm, grid_scale,max_iterations):    
    # Large blur every X iterations
    BlurLargeIter = 6
    BlurLargeIterstop = 3
    BlurRadiusLarge = 6
    BlurSmallIter = 3
    BlurSmallIterStop = 6
    BlurRadiusSmall = 3
    if (iteration % BlurLargeIter == 0) and (iteration < max_iterations -BlurLargeIterstop ):
        filter_large = create_disk_filter(0.5 * int(BlurRadiusLarge / grid_scale)).cuda()
        device_pattern = F.conv2d(device_pattern[None, None, :, :], filter_large[None, None, :, :], padding=filter_large.shape[0] // 2)[0, 0]
        device_pattern = device_pattern.to("cuda") 
    # Small blur every Y iterations
    elif (iteration % BlurSmallIter== 0) and (iteration < max_iterations - BlurSmallIterStop):
        filter_small = (create_disk_filter(BlurRadiusSmall / grid_scale)).to("cuda") 
        device_pattern = F.conv2d(device_pattern[None, None, :, :], filter_small[None, None, :, :], padding=filter_small.shape[0] // 2)[0, 0]
        device_pattern = device_pattern.to("cuda")   
    return device_pattern


# %%
def define_grid(grid, period, wavelength):
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
def replace_nan_with_1(average_final_pattern):
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
def loss_func(eps_r):
  # Global parameters dictionary.
  global params
  # Generate permittivity and permeability distributions.
  ER_t, UR_t = solver.generate_arbitrary_epsilon(eps_r, params)
  PQ_zero = int(params["PQ"][0]/2)
  ## Simulation
  outputs = solver.simulate(ER_t, UR_t, params)
  tx = outputs["tx"][:, :, :, PQ_zero, 0] # Get the zero order field by PQ_zero
  ty = outputs["ty"][:, :, :, PQ_zero, 0] 
  field = torch.unsqueeze(torch.transpose(torch.stack((tx, ty)), 0, 1), 0)
#   plot_real_part_effective_permittivity(params["Lx"],ER_t, 'refIndex.pdf')
  transmitted_field = torch.squeeze(field)
  transmitted_field = transmitted_field[0]
  FOM = outputs["T"][0,0,0,PQ_zero,PQ_zero+1]
  return 1-FOM,outputs,ER_t, UR_t,outputs

# %%
N = 1
loss = np.zeros(N + 1)
loss,outputs,ER_t, UR_t,outputs = loss_func(eps_r)
print("FOM",loss)
# torch.cuda.empty_cache()

# %%
#Main_Loop
##################################################################################################
# Number of optimization iterations.
iteration_max = 20
bin_parm = {'Min': 1,'Max': 20.0,'IterationStart': 1,'IterationHold': 3}
b_vector = generate_bvector(iteration_max, bin_parm)
opt_parm = {'Optimization': {'Iterations': iteration_max,
        'Robustness': {'StartDeviation': [-5, 0, 5],# Starting edge deviation values
            'EndDeviation': [-5, 0, 5],# Ending edge deviation values
            'Ramp':2,#Iterations over which the thresholding parameter changes
            'Weights':[.5, 1, .5]},# Gradient weight for each robustness value
            'Filter': {'BlurRadius': 3}}}
threshold_vectors, n_robustness = generate_thresh_vectors(opt_parm)
##################################################################################################
##Optimization:
var_shape = (1)
desired_phase_1_num = 180
desired_amp_1 = 1
desired_phase_2_num = 270
desired_amp_2 = 1
##################################################################################################
desired_phase_1 = desired_phase_1_num * np.ones(shape = var_shape)
desired_phase_1 = torch.tensor(desired_phase_1, dtype=torch.float32,requires_grad=True)
desired_phase_2 = desired_phase_2_num* np.ones(shape = var_shape)
desired_phase_2 = torch.tensor(desired_phase_2, dtype=torch.float32,requires_grad=True)
##################################################################################################
plt.ion()
wavelength_list = np.arange(1550, 1551, 2)#[500]
params = solver.initialize_params(wavelengths =wavelength_list,
                                  thetas = [0.0 for i in wavelength_list],
                                  phis= [0.0 for i in wavelength_list],
                                  pte= [0.0 for i in wavelength_list],#put real small numer to avoid inf
                                  ptm= [1.0 for i in wavelength_list],   
                                  erd = 17.64-.0j, # Negative imaginary part convention for loss
                                  ers = 1.69 ,
                                  PQ = [9, 9],
                                  L = [ 320.0, 2500.0],
                                  Lx = 2750.0,
                                  Ly = 1250.0,
                                  Nx = 256,
                                  eps_max =16.0-.0j )
N_x = params["Nx"]
N_y = int(np.round(params['Nx'] * params['Ly'] / params['Lx']))
eps_min = params['eps_min']
eps_max = params['eps_max']
asymmetry_x = False
asymmetry_y = False
grid = [N_x, N_x]
period = [params["Lx"]*1e9, params["Ly"]*1e9]
wavelength = wavelength_list[0]
x_grid, y_grid, dr = define_grid(grid, period, wavelength)
# Example
BlurGridLarge = 5
rand_parm = {
    'Pitch': 0.14,
    'Average': 0.5,
    'Sigma': 0.95
}
period_x = 1
period_y = 1

pattern = random_start(N_x, N_y, (period_x, period_y), rand_parm, asymmetry_x, asymmetry_y)
pattern = torch.nn.Parameter(pattern.to("cuda"))  # Wrap eps_r with Parameter for optimization
pattern.requires_grad = True
Gradient = torch.zeros_like(pattern)
Gradient = torch.nn.Parameter(Gradient.to("cuda"))  # Wrap eps_r with Parameter for optimization
Gradient.requires_grad = True
average_final_pattern = pattern.clone()
average_final_pattern = torch.nn.Parameter(average_final_pattern.to("cuda"))  # Wrap eps_r with Parameter for optimization
average_final_pattern.requires_grad = True
BlurGrid = 2;
##################################################################################################
eps_r = generate_eps_r(N_x,N_y, eps_min, eps_max, asymmetry_x, asymmetry_y)
eps_r = torch.nn.Parameter(eps_r.to("cuda"))  # Wrap eps_r with Parameter for optimization
Lx = params['Lx']
Ly = params['Ly']
eps_r_ref = generate_eps_r(N_x,N_y, eps_min, eps_min, asymmetry_x, asymmetry_y)
# Define an optimizer and data to be stored.
optimizer = optim.Adam([eps_r], lr=5e-2)
StepSize = 1#  Initial gradient step size
StepDecline = 0.99 # Multiplying factor that decreases step size each iteration
loss = np.zeros(N)
step = []
loss_mat = []
#####################################################
fig = plt.figure(figsize=(16, 6))
pos0 = [0.06, 0.1, 0.28, 0.8]
pos1 = [0.38, 0.1, 0.28, 0.8]
pos2 = [0.715, 0.1, 0.28, 0.8]
axs0 = fig.add_axes(pos0)
axs1 = fig.add_axes(pos1)
axs2 = fig.add_axes(pos2)
######################################################
# Optimize
final_pattern_mat = []
print('Optimizing...')
for iteration in range(iteration_max):
    optimizer.zero_grad()
    #First filter to enforce binarization
    FilteredPattern = density_filter_2d(pattern, BlurGridLarge)
    BinaryPattern = thresh_filter(FilteredPattern, b_vector[iteration], 0.5)
    pattern.data = replace_nan_with_1(BinaryPattern)
    eps_r = (pattern.squeeze(0).squeeze(0)*(eps_max - eps_min)) + eps_min
    loss_val,outputs,ER_t, UR_t,outputs = loss_func(eps_r)
    loss_val.backward()
    optimizer.step()
    pattern.data = enforce_symmetry(pattern, asymmetry_x, asymmetry_y)
    pattern.data = torch.clamp(pattern, min=0, max=1)
    loss_mat.append(loss_val.item())
    step.append(iteration)
    plot_real_part_effective_permittivity_and_FOM_online(loss_mat, step, Lx,Ly, ER_t,str(iteration)+'.png',
                                                         outputs)
    print(f'Iteration: {iteration+1}/{iteration_max}, FOM: {loss_val.item()}')
FilteredPattern = density_filter_2d(pattern, BlurGridLarge)
BinaryPattern = thresh_filter(FilteredPattern, b_vector[iteration], 0.5)
filtered_pattern2 = gauss_filter_2d(BinaryPattern, BlurGrid)
FinalPattern = thresh_filter(FilteredPattern, b_vector[iteration], 0.5)


# %%
def Spectrum_phase_amp(ER_t, UR_t,delta,start_wavelength,end_wavelength):
    relative_Transmission_mat = []
    relative_Phase_delay_mat = []
    wavelength_mat = []
    epsilon_mat = []
    thetas_mat = []
    stepnum = int((end_wavelength-start_wavelength)/delta)+1
    wavelengths, n_values, k_values = read_material_file('material\HamedAbr-TiO2 (1).csv')#AM_SB2_S3.txt')
    for i in range (0,stepnum):
        wavelength = start_wavelength+i*delta
        n, k, epsilon = interpolate_material(wavelengths, n_values, k_values, wavelength*1e-3)
        wavelength_mat.append(wavelength)
        epsilon_mat.append(np.conj(epsilon))
        thetas_mat.append(0.0)
    for i in range (0,stepnum):
        params = solver.initialize_params(wavelengths = [wavelength_mat[i]],   
                                      thetas = [0.0 for i in wavelength_list],
                                      phis= [0.0 for i in wavelength_list],
                                      pte= [1.0 for i in wavelength_list],#put real small numer to avoid inf
                                      ptm= [1.0 for i in wavelength_list],
                                      erd=[epsilon_mat[i]],ers=1.69, PQ=[7, 7],
                                      L = [ 300.0, 2500.0],
                                      Lx = 2000.0,
                                      Ly = 500.0,
                                      Nx = 256)
        loss_step,relative_Transmission,relative_Phase_delay,ER_t, UR_t = loss_func_spec(ER_t, UR_t,params,desired_phase_1,desired_phase_2,transmitted_field_ref)
        relative_Transmission_mat.append(relative_Transmission.cpu().detach())
        relative_Phase_delay_mat.append(relative_Phase_delay.cpu().detach())
        if i%1 == 0:
            print("simulating wavelength:",wavelength_mat[i])
#             print("simulating epsilon:",epsilon_mat[i])
    relative_Transmission_mat = [tensor.numpy() for tensor in relative_Transmission_mat]
    relative_Phase_delay_mat = [tensor.numpy() for tensor in relative_Phase_delay_mat]
    wavelength_mat = np.array(wavelength_mat)
    # Initialize empty arrays for TM and TE
    TM_phase = np.empty(len(relative_Phase_delay_mat))
    TE_phase = np.empty(len(relative_Phase_delay_mat))
    
    # Iterate through the array and extract the first and second elements
    for i, tensor_element in enumerate(relative_Phase_delay_mat):
        TM_phase[i] = tensor_element[0]
        TE_phase[i] = tensor_element[1]
    
    # Now, TM and TE contain the first and second elements, respectively
    print("TM_phase:", TM_phase)
    print("TE_phase:", TE_phase)
    ####################################################
    # Initialize empty arrays for TM and TE
    TM_amp = np.empty(len(relative_Transmission_mat))
    TE_amp = np.empty(len(relative_Transmission_mat))
    
    # Iterate through the array and extract the first and second elements
    for i, tensor_element in enumerate(relative_Transmission_mat):
        TM_amp[i] = tensor_element[0]
        TE_amp[i] = tensor_element[1]
    
    # Now, TM and TE contain the first and second elements, respectively
    print("TM_amp:", TM_amp)
    print("TE_amp:", TE_amp)
    return TM_amp,TE_amp,TM_phase,TE_phase,wavelength_mat,ER_t
############################################################################
#######Initialize grating duty cycle variable.
delta = 1*(5)+1
start_wavelength = 1530# 375
end_wavelength = 1570# 750
TM_amp,TE_amp,TM_phase,TE_phase,wavelength_mat,ER_t = Spectrum_phase_amp(ER_t, UR_t,delta,start_wavelength,end_wavelength)


# %%
def plot_polarizations(wavelength_mat, TM_amp, TM_phase, TE_amp, TE_phase,file_name):
    # Create the 2x1 subplot
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Plot TM polarization in the upper subplot
    axes[0].plot(wavelength_mat * 1e-3, TM_amp, 'k-', label='TM Relative Transmission', linewidth=4)
    axes[0].set_xlabel('$\lambda_{\mu m}$', fontsize=16)
    axes[0].set_ylabel('Transmission', fontsize=16)
    axes[0].set_title('Transmission and Phase vs Wavelength for TM Polarization', fontsize=18)
    axes[0].set_yticks([0, 1])
    axes[0].legend(loc='upper right', fontsize=14)
    axes[0].set_xlim(wavelength_mat[0]/1000,wavelength_mat[-1]/1000)
#     axes[0].grid(True)

    # Create a second y-axis on the right for TM phase delay
    ax2 = axes[0].twinx()
    ax2.plot(wavelength_mat * 1e-3, TM_phase, 'r-', label='TM Relative Phase Delay', linewidth=4)
    ax2.set_ylabel('TM Phase Delay', color='r', fontsize=16)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax2.set_yticklabels(['$-π$', '$-π/2$', '0', '$π/2$', '$π$'], fontsize=14)
    ax2.set_xlim(wavelength_mat[0]/1000,wavelength_mat[-1]/1000)

    # Plot TE polarization in the lower subplot
    axes[1].plot(wavelength_mat * 1e-3, TE_amp, 'g-', label='TE Relative Transmission', linewidth=4)
    axes[1].set_xlabel('$\lambda_{\mu m}$', fontsize=16)
    axes[1].set_ylabel('Transmission', fontsize=16)
    axes[1].set_title('Transmission and Phase vs Wavelength for TE Polarization', fontsize=18)
    axes[1].set_yticks([0, 1])
    axes[1].legend(loc='upper right', fontsize=14)
    axes[1].set_xlim(wavelength_mat[0]/1000,wavelength_mat[-1]/1000)
#     axes[1].grid(True)

    # Create a second y-axis on the right for TE phase delay
    ax3 = axes[1].twinx()
    ax3.plot(wavelength_mat * 1e-3, TE_phase, 'y-', label='TE Relative Phase Delay', linewidth=4)
    ax3.set_ylabel('TE Phase Delay', color='y', fontsize=16)
    ax3.tick_params(axis='y', labelcolor='y')
    ax3.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax3.set_yticklabels(['$-π$', '$-π/2$', '0', '$π/2$', '$π$'], fontsize=14)
    ax3.set_xlim(wavelength_mat[0]/1000,wavelength_mat[-1]/1000)

    # Save the plot as a PDF file
    plt.savefig('output/'+file_name+'.pdf', bbox_inches='tight')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

# Call the function to plot the data
plot_polarizations(wavelength_mat, TM_amp, TM_phase, TE_amp, TE_phase,'relative_transmission_phase_delay')


# %%
def apply_threshold(ER_t, threshold, eps_min, eps_max):
    # Split the complex tensor into real and imaginary parts
    ER_real = ER_t.real.to("cpu")
    ER_imag = ER_t.imag.to("cpu")

    # Apply the thresholding condition
    binary_ER_real = torch.where(ER_real < threshold, eps_min, eps_max)
    binary_ER_imag = torch.where(ER_imag < threshold, eps_min, eps_max)

    # Combine the thresholded real and imaginary parts
    binary_ER_t = binary_ER_real + 1j * binary_ER_imag

    return binary_ER_t


threshold = 8  # Set the threshold value
eps_min = params['eps_min']   # Minimum value for thresholding
eps_max = params['eps_max']  # Maximum value for thresholding
binary_ER_t = apply_threshold(ER_t, threshold, eps_min, eps_max)
plot_real_part_effective_permittivity(Lx,binary_ER_t, 'refIndex.png')
plt.figure()
plotter_iteration(loss_mat, step, 'FOM', 'output/Reflection.pdf', color='blue')
TM_amp,TE_amp,TM_phase,TE_phase,wavelength_mat,ER_t = Spectrum_phase_amp(binary_ER_t, UR_t,delta,start_wavelength,end_wavelength)
plot_polarizations(wavelength_mat, TM_amp, TM_phase, TE_amp, TE_phase,'Binary_relative_transmission_phase_delay')


# %%
import os
from PIL import Image
import imageio

# Path to the directory containing the images
image_directory = "output"

# List all image files in the directory
image_files = [file for file in os.listdir(image_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Sort the image files to ensure correct order
image_files.sort()

# Load images and append to a list
images = []
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    image = Image.open(image_path)
    images.append(image)

# Output GIF file path
if not os.path.exists('output'):
    os.makedirs('output')
gif_output_path = 'output\\output.gif'

# Save the images as a GIF using imageio
imageio.mimsave(gif_output_path, images, duration=0.5)  # You can adjust the duration between frames

print(f'GIF saved at {gif_output_path}')

# %%
import os
import cv2

# Path to the directory containing the images
image_directory = r'output\\images_opt'

# List all image files in the directory
image_files = [file for file in os.listdir(image_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Sort the image files to ensure correct order
image_files.sort()

# Get the dimensions of the first image
first_image_path = os.path.join(image_directory, image_files[0])
first_image = cv2.imread(first_image_path)
height, width, layers = first_image.shape

# Define the codec and create a VideoWriter object
if not os.path.exists('output'):
    os.makedirs('output')
video_output_path = 'output\\output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI format
out = cv2.VideoWriter(video_output_path, fourcc, 10, (width, height))  # Adjust frame rate as needed

# Write each image to the video
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    image = cv2.imread(image_path)
    out.write(image)

# Release the VideoWriter object and close the video file
out.release()

print(f'Video saved at {video_output_path}')




