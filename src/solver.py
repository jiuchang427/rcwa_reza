import torch
import numpy as np
import rcwa_utils
import tensor_utils
from src.config import Config

def initialize_params(wavelengths = [1550.0],
                      thetas = [0.0],
                      phis = [0.0],
                      pte = [0.0],
                      ptm = [1.0],
                      pixelsX = 1,
                      pixelsY = 1,
                      erd = 17.64,
                      ers = 1.69,
                      PQ = [11, 11],
                      Lx = 0.6 * 632.0,
                      Ly = 0.6 * 632.0,
                      L = [200.0, 100.0],
                      Nx = 512,
                      eps_min = 1.0,
                      eps_max = 17.64,
                      blur_radius = 100.0):
    """Initialize configuration parameters for RCWA simulation.
    
    Returns:
        Config: A configuration object containing all simulation parameters.
    """
    config = Config()
    
    # Material parameters
    config.eps_min = eps_min
    config.eps_max = eps_max
    config.erd = erd
    config.ers = ers
    
    # Geometry parameters
    config.Lx = Lx
    config.Ly = Ly
    config.Nx = Nx
    config.Ny = int(np.round(Nx * Ly / Lx))
    
    # Simulation parameters
    config.wavelengths = wavelengths
    config.thetas = thetas
    config.phis = phis
    config.pte = pte
    config.ptm = ptm
    config.PQ = PQ
    config.L = L
    
    # Additional parameters needed for solver
    config.pixelsX = pixelsX
    config.pixelsY = pixelsY
    config.blur_radius = blur_radius
    
    return config

def convert_to_tensor(data, dtype=torch.float32):
    return torch.tensor(data, dtype=dtype)

def generate_params(batchSize, pixelsX, pixelsY, wavelengths, thetas, phis, pte, ptm, erd, ers, Lx, Ly, L, PQ, Nx, eps_min, eps_max, blur_radius):
    params = {}

    # Simulation tensor shapes.
    simulation_shape = (batchSize, pixelsX, pixelsY)

    # Batch parameters (wavelength, incidence angle, and polarization).
    lam0 = convert_to_tensor(wavelengths, dtype=torch.float32) * params['nanometers']
    lam0 = lam0[:, None, None, None, None, None]
    lam0 = lam0.repeat(1, pixelsX, pixelsY, 1, 1, 1)
    params['lam0'] = lam0

    theta = convert_to_tensor(thetas, dtype=torch.float32) * params['degrees']
    theta = theta[:, None, None, None, None, None]
    theta = theta.repeat(1, pixelsX, pixelsY, 1, 1, 1)
    params['theta'] = theta

    phi = convert_to_tensor(phis, dtype=torch.float32) * params['degrees']
    phi = phi[:, None, None, None, None, None]
    phi = phi.repeat(1, pixelsX, pixelsY, 1, 1, 1)
    params['phi'] = phi

    pte = convert_to_tensor(pte, dtype=torch.complex64)
    pte = pte[:, None, None, None]
    pte = pte.repeat(1, pixelsX, pixelsY, 1)
    params['pte'] = pte

    ptm = convert_to_tensor(ptm, dtype=torch.complex64)
    ptm = ptm[:, None, None, None]
    ptm = ptm.repeat(1, pixelsX, pixelsY, 1)
    params['ptm'] = ptm

    # Device parameters.
    params['ur1'] = 1.0  # permeability in reflection region
    params['er1'] = 1.0  # permittivity in reflection region
    params['ur2'] = 1.0  # permeability in transmission region
    params['er2'] = 1.0  # permittivity in transmission region
    params['urd'] = 1.0  # permeability of device
    params['erd'] = erd  # permittivity of device
    params['urs'] = 1.0  # permeability of substrate
    params['ers'] = ers  # permittivity of substrate
    params['Lx'] = Lx * params['nanometers']  # period along x
    params['Ly'] = Ly * params['nanometers']  # period along y
    length_shape = (1, 1, 1, params['Nlay'], 1, 1)
    L = convert_to_tensor(L, dtype=torch.complex64)
    L = L[None, None, None, :, None, None]
    params['L'] = L * params['nanometers']
    params['length_min'] = 0.1
    params['length_max'] = 5.0

    # RCWA parameters.
    params['PQ'] = PQ  # number of spatial harmonics along x and y
    params['Nx'] = Nx  #

def generate_coupled_cylindrical_resonators(r_x, r_y, params):
    '''
    Generates permittivity/permeability for a unit cell comprising 4 coupled
    elliptical resonators.
    Args:
        r_x: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 4)` specifying the 
        x-axis diameters of the four cylinders.
        r_y: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 4)` specifying the 
        y-axis diameters of the four cylinders.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.
        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    '''

    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']

    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * np.ones(materials_shape)

    # Define the cartesian cross section.
    dx = Lx / Nx  # grid resolution along x
    dy = Ly / Ny  # grid resolution along y
    xa = np.linspace(0, Nx - 1, Nx) * dx  # x axis array
    xa = xa - np.mean(xa)  # center x axis at zero
    ya = np.linspace(0, Ny - 1, Ny) * dy  # y axis vector
    ya = ya - np.mean(ya)  # center y axis at zero
    y_mesh, x_mesh = np.meshgrid(ya, xa)

    # Convert to tensors and expand and tile to match the simulation shape.
    y_mesh = torch.tensor(y_mesh, dtype=torch.float32)
    y_mesh = y_mesh[None, None, None, None, :, :]
    y_mesh = y_mesh.repeat(batchSize, pixelsX, pixelsY, 1, 1, 1)
    x_mesh = torch.tensor(x_mesh, dtype=torch.float32)
    x_mesh = x_mesh[None, None, None, None, :, :]
    x_mesh = x_mesh.repeat(batchSize, pixelsX, pixelsY, 1, 1, 1)

    # Nanopost centers.
    c1_x = -Lx / 4
    c1_y = -Ly / 4
    c2_x = -Lx / 4
    c2_y = Ly / 4
    c3_x = Lx / 4
    c3_y = -Ly / 4
    c4_x = Lx / 4
    c4_y = Ly / 4

    # Clip the optimization ranges.
    r_x = params['Lx'] * torch.clamp(r_x, min=0.05, max=0.23)
    r_y = params['Ly'] * torch.clamp(r_y, min=0.05, max=0.23)
    r_x = r_x.repeat(batchSize, 1, 1, 1)
    r_y = r_y.repeat(batchSize, 1, 1, 1)
    r_x = r_x[:, :, :, None, None, None, :]
    r_y = r_y[:, :, :, None, None, None, :]

    # Calculate the nanopost boundaries.
    c1 = 1 - ((x_mesh - c1_x) / r_x[:, :, :, :, :, :, 0]) ** 2 - ((y_mesh - c1_y) / r_y[:, :, :, :, :, :, 0]) ** 2
    c2 = 1 - ((x_mesh - c2_x) / r_x[:, :, :, :, :, :, 1]) ** 2 - ((y_mesh - c2_y) / r_y[:, :, :, :, :, :, 1]) ** 2
    c3 = 1 - ((x_mesh - c3_x) / r_x[:, :, :, :, :, :, 2]) ** 2 - ((y_mesh - c3_y) / r_y[:, :, :, :, :, :, 2]) ** 2
    c4 = 1 - ((x_mesh - c4_x) / r_x[:, :, :, :, :, :, 3]) ** 2 - ((y_mesh - c4_y) / r_y[:, :, :, :, :, :, 3]) ** 2

    # Build device layer.
    ER_c1 = torch.sigmoid(params['sigmoid_coeff'] * c1)
    ER_c2 = torch.sigmoid(params['sigmoid_coeff'] * c2)
    ER_c3 = torch.sigmoid(params['sigmoid_coeff'] * c3)
    ER_c4 = torch.sigmoid(params['sigmoid_coeff'] * c4)
    ER_t = 1 + (params['erd'] - 1) * (ER_c1 + ER_c2 + ER_c3 + ER_c4)

    # Build substrate and concatenate along the layers dimension.

    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(device_shape, dtype=torch.float32)
    ER_t = torch.cat([ER_t, ER_substrate], dim=3)

    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64)
    UR_t = torch.tensor(UR, dtype=torch.float32)
    UR_t = UR_t.to(torch.complex64)

    return ER_t, UR_t
def generate_coupled_rectangular_resonators(r_x, r_y, params,wavelength_number):
    '''
    Generates permittivity/permeability for a unit cell comprising 4 coupled
    rectangular cross section scatterers.
    Args:
        r_x: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 4)` specifying the 
        x-axis widths of the four rectangles.
        r_y: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 4)` specifying the 
        y-axis widths of the four rectangles.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.
        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    '''
    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']
    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * torch.ones(materials_shape).to("cuda")
    # Define the cartesian cross section.
    dx = Lx / Nx # grid resolution along x
    dy = Ly / Ny # grid resolution along y
    xa = torch.linspace(0, Nx - 1, Nx) * dx # x axis array
    xa = xa - torch.mean(xa) # center x axis at zero
    ya = torch.linspace(0, Ny - 1, Ny) * dy # y axis vector
    ya = ya - torch.mean(ya) # center y axis at zero
    y_mesh, x_mesh = torch.meshgrid(ya,xa, indexing='ij')
    # Convert to tensors and expand and tile to match the simulation shape.
    y_mesh = y_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batchSize, pixelsX, pixelsY, 1, 1, 1)
    x_mesh = x_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batchSize, pixelsX, pixelsY, 1, 1, 1)
    ####################################################################################################
    x_mesh = x_mesh.to("cuda")
    y_mesh = y_mesh.to("cuda")

    ####################################################################################################
    # Nanopost centers.
    c1_x = -Lx / 4
    c1_y = -Ly / 4
    c2_x = -Lx / 4
    c2_y = Ly / 4
    c3_x = Lx / 4
    c3_y = -Ly / 4
    c4_x = Lx / 4
    c4_y = Ly / 4
    # Nanopost width ranges.
    r_x = params['Lx'] * torch.clip(r_x, min=0.05, max=0.23).to("cuda")
    r_y = params['Ly'] * torch.clip(r_y, min=0.05, max=0.23).to("cuda")
    r_x = r_x.repeat(batchSize, 1, 1, 1)
    r_y = r_y.repeat(batchSize, 1, 1, 1)
    r_x = r_x[:, :, :, None, None, None, :]
    r_y = r_y[:, :, :, None, None, None, :]
    # Calculate the nanopost boundaries.
    c1 = 1 - ((x_mesh - c1_x) / r_x[:, :, :, :, :, :, 0]) ** params['rectangle_power'] - ((y_mesh - c1_y) / r_y[:, :, :, :, :, :, 0]) ** params['rectangle_power']
    c2 = 1 - ((x_mesh - c2_x) / r_x[:, :, :, :, :, :, 1]) ** params['rectangle_power'] - ((y_mesh - c2_y) / r_y[:, :, :, :, :, :, 1]) ** params['rectangle_power']
    c3 = 1 - ((x_mesh - c3_x) / r_x[:, :, :, :, :, :, 2]) ** params['rectangle_power'] - ((y_mesh - c3_y) / r_y[:, :, :, :, :, :, 2]) ** params['rectangle_power']
    c4 = 1 - ((x_mesh - c4_x) / r_x[:, :, :, :, :, :, 3]) ** params['rectangle_power'] - ((y_mesh - c4_y) / r_y[:, :, :, :, :, :, 3]) ** params['rectangle_power']
    # Build device layer.
    ER_c1 = torch.sigmoid(params['sigmoid_coeff'] * c1)
    ER_c2 = torch.sigmoid(params['sigmoid_coeff'] * c2)
    ER_c3 = torch.sigmoid(params['sigmoid_coeff'] * c3)
    ER_c4 = torch.sigmoid(params['sigmoid_coeff'] * c4)
    ER_t = 1 + (params['erd'][wavelength_number]- 1) * (ER_c1 + ER_c2 + ER_c3 + ER_c4)
    # Build substrate and concatenate along the layers dimension.
    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(device_shape).to("cuda")
    ER_substrate = ER_substrate.to("cuda")
    ER_t = torch.cat([ER_t, ER_substrate], dim=3)
    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64)
    UR_t = UR.to(torch.complex64)
    return ER_t, UR_t
def generate_coupled_rectangular_resonators_opt(rx, ry, params,wavelength_number):
    '''
    Generates permittivity/permeability for a unit cell comprising 4 coupled
    rectangular cross section scatterers.
    Args:
        r_x: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 4)` specifying the 
        x-axis widths of the four rectangles.
        r_y: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 4)` specifying the 
        y-axis widths of the four rectangles.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.
        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    '''
    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']
    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * torch.ones(materials_shape).to("cuda")
    # Define the cartesian cross section.
    dx = Lx / Nx # grid resolution along x
    dy = Ly / Ny # grid resolution along y
    xa = torch.linspace(0, Nx - 1, Nx) * dx # x axis array
    xa = xa - torch.mean(xa) # center x axis at zero
    ya = torch.linspace(0, Ny - 1, Ny) * dy # y axis vector
    ya = ya - torch.mean(ya) # center y axis at zero
    y_mesh, x_mesh = torch.meshgrid(ya,xa, indexing='ij')
    # Convert to tensors and expand and tile to match the simulation shape.
    y_mesh = y_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batchSize, pixelsX, pixelsY, 1, 1, 1)
    x_mesh = x_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batchSize, pixelsX, pixelsY, 1, 1, 1)
    ####################################################################################################
    x_mesh = x_mesh.to("cuda")
    y_mesh = y_mesh.to("cuda")

    ####################################################################################################
    # Nanopost centers.
    c1_x = -Lx / 4
    c1_y = -Ly / 4
    c2_x = -Lx / 4
    c2_y = Ly / 4
    c3_x = Lx / 4
    c3_y = -Ly / 4
    c4_x = Lx / 4
    c4_y = Ly / 4
    # Nanopost width ranges.
    rx = params['Lx'] * torch.clip(rx, min=0.05, max=0.23).to("cuda")
    ry = params['Ly'] * torch.clip(ry, min=0.05, max=0.23).to("cuda")
    rx = rx.repeat(batchSize, 1, 1, 1)
    ry = ry.repeat(batchSize, 1, 1, 1)
    rx = rx[:, :, :, None, None, None, :]
    ry = ry[:, :, :, None, None, None, :]
    # Calculate the nanopost boundaries.
    c1 = 1 - ((x_mesh - c1_x) / rx[:, :, :, :, :, :, 0]) ** params['rectangle_power'] - ((y_mesh - c1_y) / ry[:, :, :, :, :, :, 0]) ** params['rectangle_power']
    c2 = 1 - ((x_mesh - c2_x) / rx[:, :, :, :, :, :, 1]) ** params['rectangle_power'] - ((y_mesh - c2_y) / ry[:, :, :, :, :, :, 1]) ** params['rectangle_power']
    c3 = 1 - ((x_mesh - c3_x) / rx[:, :, :, :, :, :, 2]) ** params['rectangle_power'] - ((y_mesh - c3_y) / ry[:, :, :, :, :, :, 2]) ** params['rectangle_power']
    c4 = 1 - ((x_mesh - c4_x) / rx[:, :, :, :, :, :, 3]) ** params['rectangle_power'] - ((y_mesh - c4_y) / ry[:, :, :, :, :, :, 3]) ** params['rectangle_power']
    # Build device layer.
    ER_c1 = torch.sigmoid(params['sigmoid_coeff'] * c1)
    ER_c2 = torch.sigmoid(params['sigmoid_coeff'] * c2)
    ER_c3 = torch.sigmoid(params['sigmoid_coeff'] * c3)
    ER_c4 = torch.sigmoid(params['sigmoid_coeff'] * c4)
    ER_t = 1 + (params['erd'][wavelength_number] - 1) * (ER_c1 + ER_c2 + ER_c3 + ER_c4)
    # Build substrate and concatenate along the layers dimension.
    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(device_shape).to("cuda")
    ER_substrate = ER_substrate.to("cuda")
    ER_t = torch.cat([ER_t, ER_substrate], dim=3)
    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64)
    UR_t = UR.to(torch.complex64)
    return ER_t, UR_t

def generate_H_resonators(L_1, L_2,L_3, L_4, L_5, params,wavelength_number):
    '''
    Generates permittivity/permeability for a unit cell comprising a single, 
    centered rectangular cross section scatterer.
    Args:
        r_x: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        x-axis widths of the rectangle.
        r_y: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        y-axis widths of the rectangle.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.
        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    '''

  # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']
#     L_5 =  torch.clamp(L_5, min=0.05, max=0.4)
    L_5 = L_5.to("cuda")

    params["L"][0,0,0,1,0,0]= L_5* params['nanometers']*1e3

    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * torch.ones(materials_shape).to("cuda")
  
    # Define the cartesian cross section.
    dx = Lx / Nx # grid resolution along x
    dy = Ly / Ny # grid resolution along y
    xa = torch.linspace(0, Nx - 1, Nx) * dx # x axis array
    xa = xa - torch.mean(xa) # center x axis at zero
    ya = torch.linspace(0, Ny - 1, Ny) * dy # y axis vector
    ya = ya - torch.mean(ya) # center y axis at zero
    y_mesh, x_mesh = torch.meshgrid(ya, xa, indexing='ij')
    ####################################################################################################
    x_mesh = x_mesh.to("cuda")
    y_mesh = y_mesh.to("cuda")
    # Limit the optimization ranges.
    L_1 =  1e-6*torch.clamp(L_1, min=0.010, max=0.500)
    L_2 =  1e-6*torch.clamp(L_2, min=0.010, max=0.500)
    L_3 =  torch.clamp(L_3, min=0.25, max=0.9)
    L_4 =  1e-6*torch.clamp(L_4, min=0.010, max=0.500)
    L_0 =  Lx / (10*L_3)-0.505*L_1

    L_1 = L_1.repeat(batchSize, 1, 1, 1)
    L_2 = L_2.repeat(batchSize, 1, 1, 1)
    L_4 = L_4.repeat(batchSize, 1, 1, 1)
    L_0 = L_0.repeat(batchSize, 1, 1, 1)

    L_1 = L_1[:, :, :, None, None, None, :]
    L_2 = L_2[:, :, :, None, None, None, :]
    L_4 = L_4[:, :, :, None, None, None, :]
    L_0 = L_0[:, :, :, None, None, None, :]

    L_1 = L_1.to("cuda")
    L_2 = L_2.to("cuda")
    L_3 = L_3.to("cuda")
    L_4 = L_4.to("cuda")
    L_0 = L_0.to("cuda")

    c1_x = -Lx / (10*L_3)


    r1 = 1 - torch.abs(((x_mesh -c1_x) / L_1[:, :, :, :, :, :, 0]) - (y_mesh / 2 / L_2[:, :, :, :, :, :, 0])) - torch.abs(((x_mesh-c1_x) / L_1[:, :, :, :, :, :, 0]) + (y_mesh / 2 / L_2[:, :, :, :, :, :, 0]))
    r2 = 1 - torch.abs(((x_mesh +c1_x) / L_1[:, :, :, :, :, :, 0]) - (y_mesh / 2 / L_2[:, :, :, :, :, :, 0])) - torch.abs(((x_mesh+c1_x) / L_1[:, :, :, :, :, :, 0]) + (y_mesh / 2 / L_2[:, :, :, :, :, :, 0]))
    r3 = 1 - torch.abs(((x_mesh) /2/ L_0[:, :, :, :, :, :, 0]) - ((y_mesh) / 2 / L_4[:, :, :, :, :, :, 0])) - torch.abs(((x_mesh)/2/ L_0[:, :, :, :, :, :, 0]) + ((y_mesh) / 2 / L_4[:, :, :, :, :, :, 0]))
  
    # Build device layer.
    ER_r1 = torch.sigmoid(params['sigmoid_coeff'] * r1)
    ER_r2 = torch.sigmoid(params['sigmoid_coeff'] * r2)
    ER_r3 = torch.sigmoid(params['sigmoid_coeff'] * r3)

    ER_t = 1 + (params['erd'][wavelength_number] - 1) * (ER_r1+ER_r2+ER_r3)
  
    # Build substrate and concatenate along the layers dimension.
    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(device_shape).to("cuda")
    ER_up = 1 * torch.ones(device_shape).to("cuda")
    ER_t = torch.cat([ER_up,ER_t, ER_substrate], dim=3)
  
    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64)
    UR_t = UR.to(torch.complex64)
    return ER_t, UR_t

def generate_H_resonators_ref(L_1, L_2,L_3, L_4, L5, params,wavelength_number):
    '''
    Generates permittivity/permeability for a unit cell comprising a single, 
    centered rectangular cross section scatterer.
    Args:
        r_x: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        x-axis widths of the rectangle.
        r_y: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        y-axis widths of the rectangle.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.
        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    '''

  # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']
  
    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * torch.ones(materials_shape).to("cuda")
  
    # Define the cartesian cross section.
    dx = Lx / Nx # grid resolution along x
    dy = Ly / Ny # grid resolution along y
    xa = torch.linspace(0, Nx - 1, Nx) * dx # x axis array
    xa = xa - torch.mean(xa) # center x axis at zero
    ya = torch.linspace(0, Ny - 1, Ny) * dy # y axis vector
    ya = ya - torch.mean(ya) # center y axis at zero
    y_mesh, x_mesh = torch.meshgrid(ya, xa, indexing='ij')
    ####################################################################################################
    x_mesh = x_mesh.to("cuda")
    y_mesh = y_mesh.to("cuda")
    # Limit the optimization ranges.
    L_1 =  1e-6*torch.clamp(L_1, min=0.010, max=0.500)
    L_2 =  1e-6*torch.clamp(L_2, min=0.010, max=0.500)
    L_1 = L_1.repeat(batchSize, 1, 1, 1)
    L_2 = L_2.repeat(batchSize, 1, 1, 1)
    L_1 = L_1[:, :, :, None, None, None, :]
    L_2 = L_2[:, :, :, None, None, None, :]
    L_1 = L_1.to("cuda")
    L_2 = L_2.to("cuda")

    r1 = 1 - torch.abs((x_mesh / 2 / L_1[:, :, :, :, :, :, 0]) - (y_mesh / 2 / L_2[:, :, :, :, :, :, 0])) - torch.abs((x_mesh / 2 / L_1[:, :, :, :, :, :, 0]) + (y_mesh / 2 / L_2[:, :, :, :, :, :, 0]))
  
    # Build device layer.
    ER_r1 = torch.sigmoid(params['sigmoid_coeff'] * r1)
    ER_t = 1 + (params['erd'][wavelength_number] - 1) * ER_r1
  
    # Build substrate and concatenate along the layers dimension.
    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(device_shape).to("cuda")
    ER_up = 1 * torch.ones(device_shape).to("cuda")
    ER_t = torch.cat([ER_up,ER_t, ER_substrate], dim=3)
  
    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64)
    UR_t = UR.to(torch.complex64)
    return ER_t, UR_t
def generate_rectangular_resonators(r_x, r_y, params,wavelength_number):
    '''
    Generates permittivity/permeability for a unit cell comprising a single, 
    centered rectangular cross section scatterer.
    Args:
        r_x: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        x-axis widths of the rectangle.
        r_y: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        y-axis widths of the rectangle.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.
        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    '''

  # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']
  
    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * torch.ones(materials_shape).to("cuda")
  
    # Define the cartesian cross section.
    dx = Lx / Nx # grid resolution along x
    dy = Ly / Ny # grid resolution along y
    xa = torch.linspace(0, Nx - 1, Nx) * dx # x axis array
    xa = xa - torch.mean(xa) # center x axis at zero
    ya = torch.linspace(0, Ny - 1, Ny) * dy # y axis vector
    ya = ya - torch.mean(ya) # center y axis at zero
    y_mesh, x_mesh = torch.meshgrid(ya, xa, indexing='ij')
    ####################################################################################################
    x_mesh = x_mesh.to("cuda")
    y_mesh = y_mesh.to("cuda")
    # Limit the optimization ranges.
    r_x =  1e-6*torch.clamp(r_x, min=0.010, max=0.500)
    r_y =  1e-6*torch.clamp(r_y, min=0.010, max=0.500)
    r_x = r_x.repeat(batchSize, 1, 1, 1)
    r_y = r_y.repeat(batchSize, 1, 1, 1)
    r_x = r_x[:, :, :, None, None, None, :]
    r_y = r_y[:, :, :, None, None, None, :]
    r_x = r_x.to("cuda")
    r_y = r_y.to("cuda")

    r1 = 1 - torch.abs((x_mesh / 2 / r_x[:, :, :, :, :, :, 0]) - (y_mesh / 2 / r_y[:, :, :, :, :, :, 0])) - torch.abs((x_mesh / 2 / r_x[:, :, :, :, :, :, 0]) + (y_mesh / 2 / r_y[:, :, :, :, :, :, 0]))
    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)

    # Build device layer.
    ER_t = 1 * torch.ones(device_shape).to("cuda")

    # Build substrate and concatenate along the layers dimension.
    ER_substrate = params['ers'] * torch.ones(device_shape).to("cuda")
    
    ER_up = 1 * torch.ones(device_shape).to("cuda")
    ER_t = torch.cat([ER_up,ER_t, ER_substrate], dim=3)  
    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64)
    UR_t = UR.to(torch.complex64)
    return ER_t, UR_t
def generate_rectangular_resonators_ref(r_x, r_y, params,wavelength_number):
    '''
    Generates permittivity/permeability for a unit cell comprising a single, 
    centered rectangular cross section scatterer.
    Args:
        r_x: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        x-axis widths of the rectangle.
        r_y: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        y-axis widths of the rectangle.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.
        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    '''

  # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']
  
    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * torch.ones(materials_shape).to("cuda")
  
    # Define the cartesian cross section.
    dx = Lx / Nx # grid resolution along x
    dy = Ly / Ny # grid resolution along y
    xa = torch.linspace(0, Nx - 1, Nx) * dx # x axis array
    xa = xa - torch.mean(xa) # center x axis at zero
    ya = torch.linspace(0, Ny - 1, Ny) * dy # y axis vector
    ya = ya - torch.mean(ya) # center y axis at zero
    y_mesh, x_mesh = torch.meshgrid(ya, xa, indexing='ij')
    ####################################################################################################
    x_mesh = x_mesh.to("cuda")
    y_mesh = y_mesh.to("cuda")
    # Limit the optimization ranges.
    r_x =  1e-6*torch.clamp(r_x, min=0.050, max=0.500)
    r_y =  1e-6*torch.clamp(r_y, min=0.050, max=0.500)
    r_x = r_x.repeat(batchSize, 1, 1, 1)
    r_y = r_y.repeat(batchSize, 1, 1, 1)
    r_x = r_x[:, :, :, None, None, None, :]
    r_y = r_y[:, :, :, None, None, None, :]
    r_x = r_x.to("cuda")
    r_y = r_y.to("cuda")

    r1 = 1 - torch.abs((x_mesh / 2 / r_x[:, :, :, :, :, :, 0]) - (y_mesh / 2 / r_y[:, :, :, :, :, :, 0])) - torch.abs((x_mesh / 2 / r_x[:, :, :, :, :, :, 0]) + (y_mesh / 2 / r_y[:, :, :, :, :, :, 0]))
    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)

    # Build device layer.
    ER_t = 1 * torch.ones(device_shape).to("cuda")

    # Build substrate and concatenate along the layers dimension.
    ER_substrate = params['ers'] * torch.ones(device_shape).to("cuda")
    
    ER_up = 1 * torch.ones(device_shape).to("cuda")
    ER_t = torch.cat([ER_up,ER_t, ER_substrate], dim=3)  
    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64)
    UR_t = UR.to(torch.complex64)
    return ER_t, UR_t
def generate_elliptical_resonators(r_x, r_y, params):
    '''
    Generates permittivity/permeability for a unit cell comprising a single, 
    centered elliptical cross section scatterer.
    Args:
        r_x: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        x-axis diameter of the ellipse.
        r_y: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        y-axis diameter of the ellipse.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.
        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    '''

    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']

    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * torch.ones(materials_shape)

    # Define the cartesian cross section.
    dx = Lx / Nx  # grid resolution along x
    dy = Ly / Ny  # grid resolution along y
    xa = torch.linspace(0, Nx - 1, Nx) * dx  # x axis array
    xa = xa - torch.mean(xa)  # center x axis at zero
    ya = torch.linspace(0, Ny - 1, Ny) * dy  # y axis vector
    ya = ya - torch.mean(ya)  # center y axis at zero
    [y_mesh, x_mesh] = torch.meshgrid(ya, xa, indexing='ij')

    # Convert to tensors and expand and tile to match the simulation shape.
    y_mesh = y_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    y_mesh = y_mesh.expand(batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    y_mesh = y_mesh.to("cuda")

    x_mesh = x_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    x_mesh = x_mesh.expand(batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    x_mesh = x_mesh.to("cuda")

    # Limit the optimization ranges.
    r_x = params['Lx'] * torch.clamp(r_x, min=0.05, max=0.46)
    r_y = params['Ly'] * torch.clamp(r_y, min=0.05, max=0.46)
    r_x = r_x.expand(batchSize, pixelsX, pixelsY, 1, 1, 1)
    r_y = r_y.expand(batchSize, pixelsX, pixelsY, 1, 1, 1)

    # Calculate the ellipse boundary.
    c1 = 1 - (x_mesh / r_x) ** 2 - (y_mesh / r_y) ** 2

    # Build device layer.
    sigmoid_coeff = torch.tensor(params['sigmoid_coeff'])
    ER_c1 = torch.sigmoid(sigmoid_coeff * c1)
    ER_t = 1 + (params['erd'] - 1) * ER_c1

    # Build substrate and concatenate along the layers dimension.
    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(device_shape)
    ER_t = torch.cat([ER_t, ER_substrate], dim=3)

    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64)
    UR_t = UR.to(torch.complex64)

    return ER_t, UR_t
def generate_cylindrical_nanoposts(duty, params):
    '''
    Generates permittivity/permeability for a unit cell comprising a single, 
    centered circular cross section scatterer.
    Args:
        duty: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, 1)` specifying the 
        duty cycle (diameter / period) of the cylindrical nanopost.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permittivity distribution of the unit cell.
        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        specifying the relative permeability distribution of the unit cell.
    '''

    # Retrieve simulation size parameters.
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']

    # Initialize relative permeability.
    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * np.ones(materials_shape)

    # Define the cartesian cross section.
    dx = params['Lx'] / Nx  # grid resolution along x
    dy = params['Ly'] / Ny  # grid resolution along y
    xa = np.linspace(0, Nx - 1, Nx) * dx  # x axis array
    xa = xa - np.mean(xa)  # center x axis at zero
    ya = np.linspace(0, Ny - 1, Ny) * dy  # y axis vector
    ya = ya - np.mean(ya)  # center y axis at zero
    [y_mesh, x_mesh] = np.meshgrid(ya, xa,indexing='ij')


     # Convert to tensors and expand and tile to match the simulation shape.     
    y_mesh = torch.from_numpy(y_mesh).to(torch.float32)
    y_mesh = y_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    y_mesh = y_mesh.repeat(batchSize, pixelsX, pixelsY, 1, 1, 1) 
    y_mesh = y_mesh.to("cuda")

    x_mesh = torch.from_numpy(x_mesh).to(torch.float32)   
    x_mesh = x_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_mesh = x_mesh.repeat(batchSize, pixelsX, pixelsY, 1, 1, 1) 
    x_mesh = x_mesh.to("cuda")

    # Build device layer.
    a = torch.clamp(duty, min=params['duty_min'], max=params['duty_max'])
    a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    a = a.repeat(1, 1, 1, 1, Nx, Ny)
    radius = 0.5 * params['Ly'] * a
    radius_cuda = radius.to("cuda")
    sigmoid_coeff_cuda =(params['sigmoid_coeff'])
    erd_cuda =(params['erd'])

    
    sigmoid_arg = (1 - (x_mesh / radius_cuda) ** 2 - (y_mesh / radius_cuda) ** 2)
    ER_t = torch.sigmoid(sigmoid_coeff_cuda* sigmoid_arg)
    ER_t = 1 + (erd_cuda - 1) * ER_t
    ers_cuda =(params['ers'])
    # Build substrate and concatenate along the layers dimension.
    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = ers_cuda * torch.ones(device_shape)
    ER_substrate = ER_substrate.to("cuda")

    ER_t = torch.cat([ER_t, ER_substrate], dim=3)
    
    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64)
    UR_t = torch.tensor(UR, dtype=torch.float32).to(torch.complex64)

    
    return ER_t, UR_t

def generate_stacked_cylindrical_nanoposts(duty, params):
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']

    UR = params['urd'] * torch.ones((batchSize, pixelsX, pixelsY, Nlay, Nx, Ny))

    dx = params['Lx'] / Nx
    dy = params['Ly'] / Ny
    xa = torch.linspace(0, Nx - 1, Nx) * dx
    xa = xa - torch.mean(xa)
    ya = torch.linspace(0, Ny - 1, Ny) * dy
    ya = ya - torch.mean(ya)
    y_mesh, x_mesh = torch.meshgrid(ya, xa, indexing='ij')

    y_mesh = y_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    y_mesh = y_mesh.expand(batchSize, pixelsX, pixelsY, Nlay - 1, Nx, Ny)
    x_mesh = x_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    x_mesh = x_mesh.expand(batchSize, pixelsX, pixelsY, Nlay - 1, Nx, Ny)

    a = torch.clamp(duty, min=params['duty_min'], max=params['duty_max'])
    a = a.unsqueeze(-1).unsqueeze(-1)
    a = a.expand(-1, -1, -1, -1, Nx, Ny)
    radius = 0.5 * params['Ly'] * a
    sigmoid_arg = (1 - (x_mesh / radius) ** 2 - (y_mesh / radius) ** 2)
    ER_t = torch.sigmoid(params['sigmoid_coeff'] * sigmoid_arg)
    ER_t = 1 + (params['erd'] - 1) * ER_t

    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(device_shape)
    ER_t = torch.cat([ER_t, ER_substrate], dim=3)

    ER_t = ER_t.to(torch.complex64)
    UR_t = UR.to(torch.complex64)

    return ER_t, UR_t
def generate_rectangular_lines(duty, params):
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']

    UR = params['urd'] * torch.ones((batchSize, pixelsX, pixelsY, Nlay, Nx, Ny))

    dx = params['Lx'] / Nx
    dy = params['Ly'] / Ny
    xa = torch.linspace(0, Nx - 1, Nx) * dx
    xa = xa - torch.mean(xa)
    ya = torch.linspace(0, Ny - 1, Ny) * dy
    ya = ya - torch.mean(ya)
    y_mesh, x_mesh = torch.meshgrid(ya, xa, indexing='ij')

    y_mesh = y_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    y_mesh = y_mesh.expand(batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    x_mesh = x_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    x_mesh = x_mesh.expand(batchSize, pixelsX, pixelsY, 1, Nx, Ny)

    a = torch.clamp(duty, min=params['duty_min'], max=params['duty_max'])
    a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    a = a.expand(-1, -1, -1, -1, Nx, Ny)
    radius = 0.5 * params['Ly'] * a
    sigmoid_arg = 1 - torch.abs(x_mesh / radius)
    ER_t = torch.sigmoid(params['sigmoid_coeff'] * sigmoid_arg)
    ER_t = 1 + (params['erd'] - 1) * ER_t

    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = params['ers'] * torch.ones(device_shape)
    ER_t = torch.cat([ER_t, ER_substrate], dim=3)

    ER_t = ER_t.to(torch.complex64)
    UR_t = UR.to(torch.complex64)

    return ER_t, UR_t
def generate_plasmonic_cylindrical_nanoposts(duty, params):
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']

    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * np.ones(materials_shape)
    
    dx = params['Lx'] / Nx
    dy = params['Ly'] / Ny
    xa = np.linspace(0, Nx - 1, Nx) * dx
    xa = xa - np.mean(xa)
    ya = np.linspace(0, Ny - 1, Ny) * dy
    ya = ya - np.mean(ya)
    [y_mesh, x_mesh] = np.meshgrid(ya, xa,indexing='ij')

    # Convert to tensors and expand and tile to match the simulation shape.     
    y_mesh = torch.from_numpy(y_mesh).to(torch.float32)
    y_mesh = y_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    y_mesh = y_mesh.repeat(batchSize, pixelsX, pixelsY, 1, 1, 1) 
    y_mesh = y_mesh.to("cuda")
    
    x_mesh = torch.from_numpy(x_mesh).to(torch.float32)   
    x_mesh = x_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_mesh = x_mesh.repeat(batchSize, pixelsX, pixelsY, 1, 1, 1) 
    x_mesh = x_mesh.to("cuda")
    
    a = torch.clamp(duty, min=params['duty_min'], max=params['duty_max'])
    a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    a = a.repeat(1, 1, 1, 1, Nx, Ny)
    radius = 0.5 * params['Lx'] * a
    radius_cuda = radius.to("cuda")
    sigmoid_coeff_cuda =(params['sigmoid_coeff'])
    erd_cuda =(params['erd'])

    sigmoid_arg = 1 - (x_mesh / radius_cuda) ** 2 - (y_mesh / radius_cuda) ** 2
    ER_t = torch.sigmoid(sigmoid_coeff_cuda* sigmoid_arg)
    ER_t = 1 + (erd_cuda) * ER_t
    ers_cuda =(params['ers'])

    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = ers_cuda * torch.ones(device_shape)
    ER_substrate = ER_substrate.to("cuda")

    ER_t = torch.cat([ER_t, ER_substrate], dim=3)
    
    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64).to("cuda")
    UR_t = torch.tensor(UR, dtype=torch.float64).to(torch.complex64).to("cuda")

    return ER_t, UR_t
def generate_plasmonic_cylindrical_nanoposts_h(var_duty_1,var_duty_2,var_duty_3,var_duty_4,
                                                                 var_duty_thickness,
                                                                 params,wavelength_number):
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']
    params["L"][0,0,0,0,0,0]= var_duty_thickness* params['nanometers']*1e3

    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * np.ones(materials_shape)
    
    dx = params['Lx'] / Nx
    dy = params['Ly'] / Ny
    xa = np.linspace(0, Nx - 1, Nx) * dx
    xa = xa - np.mean(xa)
    ya = np.linspace(0, Ny - 1, Ny) * dy
    ya = ya - np.mean(ya)
    [y_mesh, x_mesh] = np.meshgrid(ya, xa,indexing='ij')

    # Convert to tensors and expand and tile to match the simulation shape.     
    y_mesh = torch.from_numpy(y_mesh).to(torch.float32)
    y_mesh = y_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    y_mesh = y_mesh.repeat(batchSize, pixelsX, pixelsY, 1, 1, 1) 
    y_mesh = y_mesh.to("cuda")
    
    x_mesh = torch.from_numpy(x_mesh).to(torch.float32)   
    x_mesh = x_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_mesh = x_mesh.repeat(batchSize, pixelsX, pixelsY, 1, 1, 1) 
    x_mesh = x_mesh.to("cuda")
    
    a = torch.clamp(var_duty_1, min=params['duty_min'], max=params['duty_max'])
    a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    a = a.repeat(1, 1, 1, 1, Nx, Ny)
    radius_1 = 0.5 * params['Lx'] * a
    radius_cuda_1 = radius_1.to("cuda")

    
    b = torch.clamp(var_duty_2, min=params['duty_min'], max=params['duty_max'])
    b = b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    b = b.repeat(1, 1, 1, 1, Nx, Ny)
    radius_2 = 0.5 * params['Lx'] * b
    radius_cuda_2 = radius_2.to("cuda")
    
    
    c = torch.clamp(var_duty_3, min=params['duty_min'], max=params['duty_max'])
    c = c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    c = c.repeat(1, 1, 1, 1, Nx, Ny)
    radius_3 = 0.5 * params['Lx'] * c
    radius_cuda_3 = radius_3.to("cuda")
    
    d = torch.clamp(var_duty_4, min=params['duty_min'], max=params['duty_max'])
    d = d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    d = d.repeat(1, 1, 1, 1, Nx, Ny)
    radius_4 = 0.5 * params['Lx'] * d
    radius_cuda_4 = radius_4.to("cuda")  
    
    sigmoid_coeff_cuda =(params['sigmoid_coeff'])
    erd_cuda =(params['erd'][wavelength_number])
    c1_x = Lx / 4
    c1_y = Ly / 4
    sigmoid_arg_1 = 1 - ((x_mesh+c1_x) / radius_cuda_1) ** 2 - ((y_mesh-c1_y) / radius_cuda_1) ** 2
    sigmoid_arg_2 = 1 - ((x_mesh-c1_x) / radius_cuda_2) ** 2 - ((y_mesh-c1_y) / radius_cuda_2) ** 2
    sigmoid_arg_3 = 1 - ((x_mesh+c1_x) / radius_cuda_3) ** 2 - ((y_mesh+c1_y) / radius_cuda_3) ** 2
    sigmoid_arg_4 = 1 - ((x_mesh-c1_x) / radius_cuda_4) ** 2 - ((y_mesh+c1_y) / radius_cuda_4) ** 2

    ER_t_1 = torch.sigmoid(sigmoid_coeff_cuda* sigmoid_arg_1)
    ER_t_2 = torch.sigmoid(sigmoid_coeff_cuda* sigmoid_arg_2)
    ER_t_3 = torch.sigmoid(sigmoid_coeff_cuda* sigmoid_arg_3)
    ER_t_4 = torch.sigmoid(sigmoid_coeff_cuda* sigmoid_arg_4)
    
    
    ER_t = 1 + (erd_cuda) * (ER_t_1+ER_t_2+ER_t_3+ER_t_4)
    ers_cuda =(params['ers'])

    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = ers_cuda * torch.ones(device_shape)
    ER_substrate = ER_substrate.to("cuda")

    ER_t = torch.cat([ER_t, ER_substrate], dim=3)
    
    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64).to("cuda")
    UR_t = torch.tensor(UR, dtype=torch.float64).to(torch.complex64).to("cuda")

    return ER_t, UR_t
def generate_cylindrical_nanoposts_single(duty,duty2, params,wavelength_number):
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    Nlay = params['Nlay']
    Nx = params['Nx']
    Ny = params['Ny']
    Lx = params['Lx']
    Ly = params['Ly']
    params["L"][0,0,0,0,0,0]= duty2* params['nanometers']*1e3

    materials_shape = (batchSize, pixelsX, pixelsY, Nlay, Nx, Ny)
    UR = params['urd'] * np.ones(materials_shape)
    
    dx = params['Lx'] / Nx
    dy = params['Ly'] / Ny
    xa = np.linspace(0, Nx - 1, Nx) * dx
    xa = xa - np.mean(xa)
    ya = np.linspace(0, Ny - 1, Ny) * dy
    ya = ya - np.mean(ya)
    [y_mesh, x_mesh] = np.meshgrid(ya, xa,indexing='ij')

    # Convert to tensors and expand and tile to match the simulation shape.     
    y_mesh = torch.from_numpy(y_mesh).to(torch.float32)
    y_mesh = y_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    y_mesh = y_mesh.repeat(batchSize, pixelsX, pixelsY, 1, 1, 1) 
    y_mesh = y_mesh.to("cuda")
    
    x_mesh = torch.from_numpy(x_mesh).to(torch.float32)   
    x_mesh = x_mesh.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    x_mesh = x_mesh.repeat(batchSize, pixelsX, pixelsY, 1, 1, 1) 
    x_mesh = x_mesh.to("cuda")
    
    a = torch.clamp(duty, min=0.01, max=0.95)
    a = a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    a = a.repeat(1, 1, 1, 1, Nx, Ny)
    radius = 0.5 * params['Lx'] * a
    print("radius",duty*0.5 * params['Lx'])
    radius_cuda = radius.to("cuda")
    sigmoid_coeff_cuda =(params['sigmoid_coeff'])
    erd_cuda =(params['erd'][wavelength_number])
    c1_x = -Lx / 4
    c1_y = -Ly / 4
    sigmoid_arg = 1 - ((x_mesh) / radius_cuda) ** 2 - ((y_mesh) / radius_cuda) ** 2
    ER_t = torch.sigmoid(sigmoid_coeff_cuda* sigmoid_arg)
    ER_t = 1 + (erd_cuda) * ER_t
    ers_cuda =(params['ers'])

    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = ers_cuda * torch.ones(device_shape)
    ER_substrate = ER_substrate.to("cuda")

    ER_t = torch.cat([ER_t, ER_substrate], dim=3)
    
    # Cast to complex for subsequent calculations.
    ER_t = ER_t.to(torch.complex64).to("cuda")
    UR_t = torch.tensor(UR, dtype=torch.float64).to(torch.complex64).to("cuda")

    return ER_t, UR_t

def generate_arbitrary_epsilon(eps_r, config):
    batchSize = 1  # handle batching in the Config class if needed
    pixelsX = config.pixelsX
    pixelsY = config.pixelsY
    Nlay = len(config.L)
    Nx = config.Nx
    Ny = config.Ny

    UR = config.erd * torch.ones((batchSize, pixelsX, pixelsY, Nlay, Nx, Ny))
    UR = UR.to("cuda")

    ER_t = torch.clamp(torch.real(eps_r), min=config.eps_min, max=np.real(config.eps_max))
    ER_t = ER_t.repeat(batchSize, 1, 1, 1, 1, 1)
    ER_t = ER_t.to("cuda")
    device_shape = (batchSize, pixelsX, pixelsY, 1, Nx, Ny)
    ER_substrate = config.ers * torch.ones(device_shape, dtype=torch.float32)
    ER_substrate = ER_substrate.to("cuda")
    ER_t = torch.cat([ER_t, ER_substrate], dim=3)

    # Add substrate permeability layer
    UR_substrate = torch.ones(device_shape, dtype=torch.float32)  # Non-magnetic substrate has permeability = 1.0
    UR_substrate = UR_substrate.to("cuda")
    UR_t = torch.cat([UR, UR_substrate], dim=3)

    ER_t = ER_t.to(torch.complex64)
    UR_t = UR_t.clone().detach().to(torch.float32)
    UR_t = UR_t.to(torch.complex64)
    ER_t = ER_t.to("cuda")
    UR_t = UR_t.to("cuda")
    return ER_t, UR_t

def make_propagator(params, f):
    batchSize = params['batchSize']
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    upsample = params['upsample']

    k = 2 * np.pi / params['lam0'][:, 0, 0, 0, 0, 0]
    k = k[:, np.newaxis, np.newaxis]
    samp = upsample * pixelsX
    k = torch.tile(k, (1, 2 * samp - 1, 2 * samp - 1))
    k = k.to(torch.complex64)
    k_xlist_pos = 2 * np.pi * np.linspace(0, 1 / (2 * params['Lx'] / upsample), samp)
    front = k_xlist_pos[-(samp - 1):]
    front = -front[::-1]
    k_xlist = np.hstack((front, k_xlist_pos))
    k_x = np.kron(k_xlist, np.ones((2 * samp - 1, 1)))
    k_x = k_x[np.newaxis, :, :]
    k_y = np.transpose(k_x, axes=[0, 2, 1])
    k_x = torch.tensor(k_x, dtype=torch.complex64)
    k_x = k_x.repeat(batchSize, 1, 1)
    k_y = torch.tensor(k_y, dtype=torch.complex64)
    k_y = k_y.repeat(batchSize, 1, 1)
    k_z_arg = torch.square(k) - (torch.square(k_x) + torch.square(k_y))
    k_z = torch.sqrt(k_z_arg)
    propagator_arg = 1j * k_z * f
    propagator = torch.exp(propagator_arg)

    kx_limit = 2 * np.pi * (((1 / (pixelsX * params['Lx'])) * f) ** 2 + 1) ** (-0.5) / params['lam0'][:, 0, 0, 0, 0, 0]
    kx_limit = kx_limit.to(torch.complex64)
    ky_limit = kx_limit
    kx_limit = kx_limit[:, np.newaxis, np.newaxis]
    ky_limit = ky_limit[:, np.newaxis, np.newaxis]

    ellipse_kx = (torch.square(k_x / kx_limit) + torch.square(k_y / k)).cpu().numpy() <= 1
    ellipse_ky = (torch.square(k_x / k) + torch.square(k_y / ky_limit)).cpu().numpy() <= 1
    propagator = propagator * torch.tensor(ellipse_kx, dtype=torch.complex64) * torch.tensor(ellipse_ky, dtype=torch.complex64)

    return propagator
def propagate(field, propagator, upsample):
    batchSize, _, _ = field.shape
    _, n, _ = propagator.shape

    field = field.permute(2, 0, 1)  # Transpose to put batch parameter last
    field_real = field.real
    field_imag = field.imag
    field_real = torch.nn.functional.interpolate(field_real, size=(n, n), mode='nearest')
    field_imag = torch.nn.functional.interpolate(field_imag, size=(n, n), mode='nearest')
    field = torch.complex(field_real, field_imag)
    field = torch.nn.functional.pad(field, (0, 2 * n - 1 - n, 0, 2 * n - 1 - n))
    field = field.permute(1, 2, 0)

    field_freq = torch.fft.fftshift(torch.fft.fft2(field), dim=(1, 2))
    field_filtered = torch.fft.ifftshift(field_freq * propagator, dim=(1, 2))
    out = torch.fft.ifft2(field_filtered)

    out = out.permute(2, 0, 1)
    out = torch.nn.functional.pad(out, (0, n - out.shape[1], 0, n - out.shape[2]))
    out = out.permute(1, 2, 0)

    return out
def define_input_fields(params):
    pixelsX = params['pixelsX']
    pixelsY = params['pixelsY']
    dx = params['Lx']
    dy = params['Ly']
    xa = np.linspace(0, pixelsX - 1, pixelsX) * dx
    xa = xa - np.mean(xa)
    ya = np.linspace(0, pixelsY - 1, pixelsY) * dy
    ya = ya - np.mean(ya)
    y_mesh, x_mesh = np.meshgrid(ya, xa)
    x_mesh = x_mesh[np.newaxis, :, :]
    y_mesh = y_mesh[np.newaxis, :, :]

    lam_phase_test = params['lam0'][:, 0, 0, 0, 0, 0]
    lam_phase_test = lam_phase_test[:, np.newaxis, np.newaxis]
    theta_phase_test = params['theta'][:, 0, 0, 0, 0, 0]
    theta_phase_test = theta_phase_test[:, np.newaxis, np.newaxis]

    phase_def = 2 * np.pi * torch.sin(theta_phase_test) * x_mesh / lam_phase_test
    phase_def = torch.complex(phase_def, torch.zeros_like(phase_def))
    phase_def = phase_def.to(torch.complex64)

    return torch.exp(1j * phase_def)
def simulate(ER_t, UR_t, config):
    '''
    Calculates the transmission/reflection coefficients for a unit cell with a
    given permittivity/permeability distribution and the batch of input conditions 
    (e.g., wavelengths, wavevectors, polarizations) for a fixed real space grid
    and number of Fourier harmonics.
    Args:
        ER_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        and dtype `torch.cfloat` specifying the relative permittivity distribution
        of the unit cell.
        UR_t: A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nlayer, Nx, Ny)`
        and dtype `torch.cfloat` specifying the relative permeability distribution
        of the unit cell.
        config: A `Config` object containing simulation and optimization settings.
    Returns:
        outputs: A `dict` containing the keys {'rx', 'ry', 'rz', 'R', 'ref', 
        'tx', 'ty', 'tz', 'T', 'TRN'} corresponding to the computed reflection/tranmission
        coefficients and powers.
    '''
    batchSize = 1  # We'll handle batching in the Config class if needed
    pixelsX = config.pixelsX
    pixelsY = config.pixelsY
    Nlay = len(config.L)
    PQ = config.PQ
  
    ### Step 3: Build convolution matrices for the permittivity and permeability ###
    ERC = rcwa_utils.convmat(ER_t, PQ[0], PQ[1]).to("cuda")
    URC = rcwa_utils.convmat(UR_t, PQ[0], PQ[1]).to("cuda")

    ### Step 4: Wave vector expansion ###
    I = np.eye(np.prod(PQ), dtype=complex)
    I = torch.from_numpy(I).to(torch.complex64).to("cuda")
    I = I.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    I = I.repeat(batchSize, pixelsX, pixelsY, Nlay, 1, 1)
    Z = np.zeros((np.prod(PQ), np.prod(PQ)), dtype = complex)
    Z = torch.from_numpy(Z).to(torch.complex64).to("cuda") 
    Z = Z.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    Z = Z.repeat(batchSize, pixelsX, pixelsY, Nlay, 1, 1)
    n1 = np.sqrt(config.er1)
    n2 = np.sqrt(config.er2)
  
    # Use tensors directly from config
    k0 = 2 * np.pi / config.lam0
    kinc_x0 = n1 * torch.sin(config.theta) * torch.cos(config.phi)
    kinc_y0 = n1 * torch.sin(config.theta) * torch.sin(config.phi)
    kinc_z0 = n1 * torch.cos(config.theta)
    kinc_z0 = kinc_z0[:, :, :, 0, :, :]
    
    # Unit vectors
    T1 = np.transpose([2 * np.pi / config.Lx, 0])
    T2 = np.transpose([0, 2 * np.pi / config.Ly])
    p_max = np.floor(PQ[0] / 2.0)
    q_max = np.floor(PQ[1] / 2.0)
    p = torch.linspace(-p_max, p_max, PQ[0]).type(torch.complex64).to("cuda") # indices along T1
    p = p.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    p = p.repeat(1, pixelsX, pixelsY, Nlay, 1, 1)
    q = torch.linspace(-q_max, q_max, PQ[1]).type(torch.complex64).to("cuda") # indices along T2
    q = q.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    q = q.repeat(1, pixelsX, pixelsY, Nlay, 1, 1)
    
    # Build Kx and Ky matrices
    kx_zeros = torch.zeros(PQ[1]).type(torch.complex64).to("cuda")
    kx_zeros = kx_zeros.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    ky_zeros = torch.zeros(PQ[0]).type(torch.complex64).to("cuda")
    ky_zeros = ky_zeros.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    kx = kinc_x0 - 2 * np.pi * p / (k0 * config.Lx) - kx_zeros
    ky = kinc_y0 - 2 * np.pi * q / (k0 * config.Ly) - ky_zeros
    
    kx_T = kx.permute(0, 1, 2, 3, 5, 4)
    KX = kx_T.reshape(batchSize, pixelsX, pixelsY, Nlay, np.prod(PQ))
    KX = torch.diag_embed(KX).to("cuda")
    
    ky_T = ky.permute(0, 1, 2, 3, 5, 4)
    KY = ky_T.reshape(batchSize, pixelsX, pixelsY, Nlay, np.prod(PQ))
    KY = torch.diag_embed(KY).to("cuda")
    
    KZref = torch.matmul(torch.conj(config.ur1 * I), torch.conj(config.er1 * I))
    KZref = KZref - torch.matmul(KX, KX) - torch.matmul(KY, KY)
    KZref = torch.sqrt(KZref).to("cuda")
    KZref = -torch.conj(KZref).to("cuda")
    
    KZtrn = torch.matmul(torch.conj(config.ur2 * I), torch.conj(config.er2 * I))
    KZtrn = KZtrn - torch.matmul(KX, KX) - torch.matmul(KY, KY)
    KZtrn = torch.sqrt(KZtrn).to("cuda")
    KZtrn = torch.conj(KZtrn).to("cuda")
    
    # Step 5: Free Space
    KZ = I - torch.matmul(KX, KX) - torch.matmul(KY, KY)
    KZ = torch.sqrt(KZ).to("cuda")
    KZ = torch.conj(KZ).to("cuda")
    
    Q_free_00 = torch.matmul(KX, KY).to("cuda")
    Q_free_01 = I - torch.matmul(KX, KX).to("cuda")
    
    Q_free_10 = torch.matmul(KY, KY) - I
    Q_free_11 = -torch.matmul(KY, KX)
    Q_free_row0 = torch.cat([Q_free_00, Q_free_01], dim = 5).to("cuda")
    Q_free_row1 = torch.cat([Q_free_10, Q_free_11], dim = 5).to("cuda")
    Q_free = torch.cat([Q_free_row0, Q_free_row1], dim = 4).to("cuda")
    
    W0_row0 = torch.cat([I, Z], dim = 5).to("cuda")
    W0_row1 = torch.cat([Z, I], dim = 5).to("cuda")
    W0 = torch.cat([W0_row0, W0_row1], dim = 4).to("cuda")
    
    LAM_free_row0 = torch.cat([1j * KZ, Z], dim = 5).to("cuda")
    LAM_free_row1 = torch.cat([Z, 1j * KZ], dim = 5).to("cuda")
    LAM_free = torch.cat([LAM_free_row0, LAM_free_row1], dim = 4)
    
    V0 = torch.matmul(Q_free, torch.inverse(LAM_free)).to("cuda")
    
    # Step 6: Initialize Global Scattering Matrix
    SG = dict({})
    SG_S11 = torch.zeros((2 * np.prod(PQ), 2 * np.prod(PQ)), dtype = torch.complex64).to("cuda")
    SG['S11'] = tensor_utils.expand_and_tile_torch(SG_S11, batchSize, pixelsX, pixelsY)
    
    SG_S12 = torch.eye(2 * np.prod(PQ), dtype = torch.complex64).to("cuda")
    SG['S12'] = tensor_utils.expand_and_tile_torch(SG_S12, batchSize, pixelsX, pixelsY)
    
    SG_S21 = torch.eye(2 * np.prod(PQ), dtype = torch.complex64).to("cuda")
    SG['S21'] = tensor_utils.expand_and_tile_torch(SG_S21, batchSize, pixelsX, pixelsY)
    
    SG_S22 = torch.zeros((2 * np.prod(PQ), 2 * np.prod(PQ)), dtype = torch.complex64).to("cuda")
    SG['S22'] = tensor_utils.expand_and_tile_torch(SG_S22, batchSize, pixelsX, pixelsY)
    
    # Step 7: Calculate eigenmodes
    # Build the eigenvalue problem.
    KX = KX.to("cuda")
    ERC = ERC.to("cuda")
    KY = KY.to("cuda")
    URC = URC.to("cuda")
    P_00 = torch.matmul(KX, torch.inverse(ERC))
    P_00 = torch.matmul(P_00, KY)
    
    P_01 = torch.matmul(KX, torch.inverse(ERC))

    P_01 = torch.matmul(P_01, KX)
    P_01 = URC - P_01

    P_10 = torch.matmul(KY, torch.inverse(ERC))
    P_10 = torch.matmul(P_10, KY) - URC
    
    P_11 = torch.matmul(-KY, torch.inverse(ERC))
    P_11 = torch.matmul(P_11, KX)
    
    P_row0 = torch.cat([P_00, P_01], dim = 5)
    P_row1 = torch.cat([P_10, P_11], dim = 5)
    P = torch.cat([P_row0, P_row1], dim = 4)
    
    Q_00 = torch.matmul(KX, torch.inverse(URC))
    Q_00 = torch.matmul(Q_00, KY)
    
    Q_01 = torch.matmul(KX, torch.inverse(URC))
    Q_01 = torch.matmul(Q_01, KX)
    Q_01 = ERC - Q_01
    
    Q_10 = torch.matmul(KY, torch.inverse(URC))
    Q_10 = torch.matmul(Q_10, KY) - ERC
    
    Q_11 = torch.matmul(-KY, torch.inverse(URC))
    Q_11 = torch.matmul(Q_11, KX)
    
    Q_row0 = torch.cat([Q_00, Q_01], dim = 5)
    Q_row1 = torch.cat([Q_10, Q_11], dim = 5)
    Q = torch.cat([Q_row0, Q_row1], dim = 4)
#     print("Q",Q)
#     print("P",P)

    # Compute eigenmodes for the layers in each pixel for the whole batch.
    OMEGA_SQ = torch.matmul(P, Q).to("cuda")# eigenvalues==>LAM, eigenvectors==>W
    
    LAM, W = tensor_utils.EigGeneral.apply(OMEGA_SQ)
#     print("LAM",LAM)
    del OMEGA_SQ
#     torch.cuda.empty_cache()
    LAM = torch.sqrt(LAM).to("cuda")
    LAM = torch.diag_embed(LAM).to("cuda")

    V = torch.matmul(Q, W).to("cuda")
    V = torch.matmul(V, torch.inverse(LAM)).to("cuda")
#     print("V",V)
    # Scattering matrices for the layers in each pixel for the whole batch.
    W_inv = torch.inverse(W).to("cuda")
    V_inv = torch.inverse(V).to("cuda")
    A = torch.matmul(W_inv, W0) + torch.matmul(V_inv, V0).to("cuda")
    B = torch.matmul(W_inv, W0) - torch.matmul(V_inv, V0).to("cuda")
#     print("-LAM",LAM.device)

    L_cuda = config.L.to("cuda")

    X = torch.matrix_exp(-LAM * k0 * L_cuda)
    S = dict({})
    A_inv = torch.inverse(A).to("cuda")
    S11_left = torch.matmul(X, B).to("cuda")
    S11_left = torch.matmul(S11_left, A_inv).to("cuda")
    S11_left = torch.matmul(S11_left, X).to("cuda")
    S11_left = torch.matmul(S11_left, B).to("cuda")
    S11_left = A - S11_left
    S11_left = torch.inverse(S11_left).to("cuda")
    
    S11_right = torch.matmul(X, B).to("cuda")
    S11_right = torch.matmul(S11_right, A_inv).to("cuda")
    S11_right = torch.matmul(S11_right, X).to("cuda")
    S11_right = torch.matmul(S11_right, A).to("cuda")
    S11_right = S11_right - B
    S['S11'] = torch.matmul(S11_left, S11_right).to("cuda")
    
    S12_right = torch.matmul(B, A_inv).to("cuda")
    S12_right = torch.matmul(S12_right, B).to("cuda")
    S12_right = A - S12_right
    S12_left = torch.matmul(S11_left, X).to("cuda")
    S['S12'] = torch.matmul(S12_left, S12_right).to("cuda")
    
    S['S21'] = S['S12']
    S['S22'] = S['S11']
    
    # Update the global scattering matrices.
    for l in range(Nlay):
        S_layer = dict({})
        S_layer['S11'] = S['S11'][:, :, :, l, :, :]
        S_layer['S12'] = S['S12'][:, :, :, l, :, :]
        S_layer['S21'] = S['S21'][:, :, :, l, :, :]
        S_layer['S22'] = S['S22'][:, :, :, l, :, :]
        SG = rcwa_utils.redheffer_star_product(SG, S_layer) # You may need to replace this with an appropriate     function from your rcwa_utils module  
    # Step 8: Reflection side
    # Eliminate layer dimension for tensors as they are unchanging on this dimension.
    KX = KX[:, :, :, 0, :, :]
    KY = KY[:, :, :, 0, :, :]
    KZref = KZref[:, :, :, 0, :, :]
    KZtrn = KZtrn[:, :, :, 0, :, :]
    Z = Z[:, :, :, 0, :, :]
    I = I[:, :, :, 0, :, :]
    W0 = W0[:, :, :, 0, :, :]
    V0 = V0[:, :, :, 0, :, :]
    Q_ref_00 = torch.matmul(KX, KY)
    Q_ref_01 = config.ur1 * config.er1 * I - torch.matmul(KX, KX)
    Q_ref_10 = torch.matmul(KY, KY) - config.ur1 * config.er1 * I
    Q_ref_11 = -torch.matmul(KY, KX)
    Q_ref_row0 = torch.cat([Q_ref_00, Q_ref_01], dim = 4)
    Q_ref_row1 = torch.cat([Q_ref_10, Q_ref_11], dim = 4)
    Q_ref = torch.cat([Q_ref_row0, Q_ref_row1], dim = 3)
    
    W_ref_row0 = torch.cat([I, Z], dim = 4)
    W_ref_row1 = torch.cat([Z, I], dim = 4)
    W_ref = torch.cat([W_ref_row0, W_ref_row1], dim = 3)
    
    LAM_ref_row0 = torch.cat([-1j * KZref, Z], dim = 4)
    LAM_ref_row1 = torch.cat([Z, -1j * KZref], dim = 4)
    LAM_ref = torch.cat([LAM_ref_row0, LAM_ref_row1], dim = 3)
    
    V_ref = torch.matmul(Q_ref, torch.inverse(LAM_ref))
    
    W0_inv = torch.inverse(W0)

    V0_inv = torch.inverse(V0)
    A_ref = torch.matmul(W0_inv, W_ref) + torch.matmul(V0_inv, V_ref)
    A_ref_inv = torch.inverse(A_ref)
    B_ref = torch.matmul(W0_inv, W_ref) - torch.matmul(V0_inv, V_ref)
    
    SR = dict({})
    SR['S11'] = torch.matmul(-A_ref_inv, B_ref)
    SR['S12'] = 2 * A_ref_inv
    SR_S21 = torch.matmul(B_ref, A_ref_inv)
    SR_S21 = torch.matmul(SR_S21, B_ref)
    SR['S21'] = 0.5 * (A_ref - SR_S21)
    SR['S22'] = torch.matmul(B_ref, A_ref_inv)
    
    # Step 9: Transmission side
    Q_trn_00 = torch.matmul(KX, KY)
    Q_trn_01 = config.ur2 * config.er2 * I - torch.matmul(KX, KX)
    Q_trn_10 = torch.matmul(KY, KY) - config.ur2 * config.er2 * I
    Q_trn_11 = -torch.matmul(KY, KX)
    Q_trn_row0 = torch.cat([Q_trn_00, Q_trn_01], dim = 4)
    Q_trn_row1 = torch.cat([Q_trn_10, Q_trn_11], dim = 4)
    Q_trn = torch.cat([Q_trn_row0, Q_trn_row1], dim = 3)
    
    W_trn_row0 = torch.cat([I, Z], dim = 4)
    W_trn_row1 = torch.cat([Z, I], dim = 4)
    W_trn = torch.cat([W_trn_row0, W_trn_row1], dim = 3)
    
    LAM_trn_row0 = torch.cat([1j * KZtrn, Z], dim = 4)
    LAM_trn_row1 = torch.cat([Z, 1j * KZtrn], dim = 4)
    LAM_trn = torch.cat([LAM_trn_row0, LAM_trn_row1], dim=3)
    V_trn = torch.matmul(Q_trn, torch.inverse(LAM_trn))
    
    W0_inv = torch.inverse(W0)
    V0_inv = torch.inverse(V0)
    A_trn = torch.matmul(W0_inv, W_trn) + torch.matmul(V0_inv, V_trn)
    A_trn_inv = torch.inverse(A_trn)
    B_trn = torch.matmul(W0_inv, W_trn) - torch.matmul(V0_inv, V_trn)
    
    ST = {}
    ST['S11'] = torch.matmul(B_trn, A_trn_inv)
    ST_S12 = torch.matmul(B_trn, A_trn_inv)
    ST_S12 = torch.matmul(ST_S12, B_trn)
    ST['S12'] = 0.5 * (A_trn - ST_S12)
    ST['S21'] = 2 * A_trn_inv
    ST['S22'] = torch.matmul(-A_trn_inv, B_trn)
    # Step 10: Compute global scattering matrix
    # You need to implement the redheffer_star_product function in PyTorch
    SG = rcwa_utils.redheffer_star_product(SR, SG)
    SG = rcwa_utils.redheffer_star_product(SG, ST)
    
    # Step 11: Compute source parameters
    # Compute mode coefficients of the source.
    delta = np.zeros((batchSize, pixelsX, pixelsY, np.prod(PQ)))
    delta[:, :, :, int(np.prod(PQ) / 2.0)] = 1
    delta = torch.from_numpy(delta).to("cuda")
    
    # Incident wavevector.
    kinc_x0_pol = torch.real(kinc_x0[:, :, :, 0, 0]).to("cuda")
    kinc_y0_pol = torch.real(kinc_y0[:, :, :, 0, 0]).to("cuda")
    kinc_z0_pol = torch.real(kinc_z0[:, :, :, 0]).to("cuda")
    kinc_pol = torch.cat([kinc_x0_pol, kinc_y0_pol, kinc_z0_pol], dim=3).to("cuda")
#     print("kinc_pol",kinc_pol)

    # Calculate TE and TM polarization unit vectors.
    firstPol = True
    for pol in range(batchSize):
        if (kinc_pol[pol, 0, 0, 0] == 0.0 and kinc_pol[pol, 0, 0, 1] == 0.0):
            ate_pol = np.zeros((1, pixelsX, pixelsY, 3))
            ate_pol[:, :, :, 1] = 1
            ate_pol = (torch.from_numpy(ate_pol).float()).to("cuda")
        else:
            # Calculation of `ate` for oblique incidence.
            n_hat = np.zeros((1, pixelsX, pixelsY, 3))
            n_hat[:, :, :, 0] = 1
            n_hat = (torch.from_numpy(n_hat).float()).to("cuda")
            kinc_pol_iter = kinc_pol[pol, :, :, :].unsqueeze(0)
            ate_cross = torch.linalg.cross(n_hat, kinc_pol_iter, dim=-1)
            ate_pol = ate_cross / torch.norm(ate_cross, dim=3, keepdim=True)
    
        if firstPol:
            ate = ate_pol
            firstPol = False
        else:
            ate = torch.cat([ate, ate_pol], dim=0).to("cuda")
    
    atm_cross = torch.linalg.cross(kinc_pol, ate, dim=-1)
    atm = atm_cross / torch.norm(atm_cross, dim=3, keepdim=True)
    ate = ate.to(dtype=torch.complex64)
    atm = atm.to(dtype=torch.complex64)
    
    # Decompose the TE and TM polarization into x and y components.
    pte_cuda = config.pte.to("cuda")
    ptm_cuda = config.ptm.to("cuda")
    EP = pte_cuda * ate + ptm_cuda * atm
    EP_x = (EP[:, :, :, 0].unsqueeze(-1)).to("cuda")
    EP_y = (EP[:, :, :, 1].unsqueeze(-1)).to("cuda")
    
    esrc_x = EP_x * delta
    esrc_y = EP_y * delta
    esrc = torch.cat([esrc_x, esrc_y], dim=3).to("cuda")
    esrc = esrc.unsqueeze(-1)
    
    W_ref_inv = torch.inverse(W_ref).to("cuda")
    # Step 12: Compute reflected and transmitted fields
    W_ref_inv = W_ref_inv.to(torch.complex64)
    esrc = esrc.to(torch.complex64)
    csrc = torch.matmul(W_ref_inv, esrc).to("cuda")
    
    # Compute transmission and reflection mode coefficients.
    cref = torch.matmul(SG['S11'], csrc).to("cuda")

    ctrn = torch.matmul(SG['S21'], csrc).to("cuda")
    eref = torch.matmul(W_ref, cref).to("cuda")

    etrn = torch.matmul(W_trn, ctrn).to("cuda")
    
    rx = eref[:, :, :, 0 : np.prod(PQ), :]
    ry = eref[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]
    tx = etrn[:, :, :, 0 : np.prod(PQ), :]
    ty = etrn[:, :, :, np.prod(PQ) : 2 * np.prod(PQ), :]
    
    # Compute longitudinal components.
    KZref_inv = torch.inverse(KZref).to("cuda")
    KZtrn_inv = torch.inverse(KZtrn).to("cuda")
    rz = torch.matmul(KX, rx) + torch.matmul(KY, ry)
    rz = torch.matmul(-KZref_inv, rz).to("cuda")
    tz = torch.matmul(KX, tx) + torch.matmul(KY, ty)
    tz = torch.matmul(-KZtrn_inv, tz).to("cuda")
    
    # Step 13: Compute diffraction efficiencies
    rx2 = torch.real(rx) ** 2 + torch.imag(rx) ** 2
    ry2 = torch.real(ry) ** 2 + torch.imag(ry) ** 2
    rz2 = torch.real(rz) ** 2 + torch.imag(rz) ** 2
    R2 = rx2 + ry2 + rz2
    R = torch.real(-KZref / config.ur1) / torch.real(kinc_z0 / config.ur1)
    R = torch.matmul(R, R2).to("cuda")
    R = R.view(batchSize, pixelsX, pixelsY, PQ[0], PQ[1])
    REF = torch.sum(R, dim=[3, 4])

    tx2 = torch.real(tx) ** 2 + torch.imag(tx) ** 2
    ty2 = torch.real(ty) ** 2 + torch.imag(ty) ** 2
    tz2 = torch.real(tz) ** 2 + torch.imag(tz) ** 2
    T2 = tx2 + ty2 + tz2
    T = torch.real(KZtrn / config.ur2) / torch.real(kinc_z0 / config.ur2)
    T = torch.matmul(T, T2)
    T = T.view(batchSize, pixelsX, pixelsY, PQ[0], PQ[1])
    TRN = torch.sum(T, dim=[3, 4])
    
    # Store the transmission/reflection coefficients and powers in a dictionary.
    outputs = dict()
    outputs['rx'] = rx
    outputs['ry'] = ry
    outputs['rz'] = rz
    outputs['R'] = R
    outputs['REF'] = REF
    outputs['tx'] = tx
    outputs['ty'] = ty
    outputs['tz'] = tz
    outputs['T'] = T
    outputs['TRN'] = TRN
    
    return outputs
