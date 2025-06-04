import torch
import torch.nn.functional as F
import numpy as np

def convolve_density_with_blur(density, blur):
    '''
    This function computes the convolution of two inputs to return a blurred
    density function.
    Args:
        density: A `torch.Tensor` of dtype `torch.float32` and shape `(1, pixelsX, 
        pixelsY, Nlayers - 1, Nx, Ny)` specifying a density function with values
        in the range from 0 to 1 on the Nx and Ny dimensions.
        blur: A `torch.Tensor` of dtype `torch.float32` and shape `(1, pixelsX, 
        pixelsY, Nlayers - 1, Nx, Ny)` specifying a blur function on the Nx and
        Ny dimensions.
    Returns:
        A `torch.Tensor` of dtype `torch.float32` and shape `(1, pixelsX, pixelsY, 
        Nlayers - 1, Nx, Ny)` specifying the blurred density.
    '''

    _, _, _, _, Nx, Ny = density.shape

    # Padding to accommodate linear convolution.
    paddings = (Ny // 2, Ny // 2, Nx // 2, Nx // 2, 0, 0, 0, 0, 0, 0)
    density_padded = F.pad(density, pad=paddings)
    density_padded = torch.stack((density_padded, torch.zeros_like(density_padded)), dim=-1)

    blur_padded = F.pad(blur, pad=paddings)
    blur_padded = torch.stack((blur_padded, torch.zeros_like(blur_padded)), dim=-1)

    # Perform the convolution in the Fourier domain and return the image.
    convolved = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(density_padded) * torch.fft.fft2(blur_padded)), dim=(4, 5))
    x_low = Nx // 2
    x_high = x_low + Nx
    y_low = Ny // 2
    y_high = y_low + Ny
    convolved_cropped = convolved[:, :, :, :, x_low : x_high, y_low : y_high]

    return torch.abs(convolved_cropped)

def blur_unit_cell(eps_r, params):
    '''
    This function blurs a unit cell to remove high spatial frequency features.
    Args:
        eps_r: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, Nlayer - 1, Nx, Ny)`
        and dtype `torch.float32` specifying the permittivity at each point in the 
        unit cell grid.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        A `torch.Tensor` of dtype `torch.float32` and shape `(1, pixelsX, pixelsY, 
        Nlayers - 1, Nx, Ny)` specifying the blurred real space permittivity
        on a Cartesian grid.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eps_r = eps_r.to(device)

    # Define the cartesian cross section.
    dx = params['Lx'] / params['Nx'] # grid resolution along x
    dy = params['Ly'] / params['Ny'] # grid resolution along y
    xa = np.linspace(0, params['Nx'] - 1, params['Nx']) * dx # x axis array
    xa = xa - np.mean(xa) # center x axis at zero
    ya = np.linspace(0, params['Ny'] - 1, params['Ny']) * dy # y axis vector
    ya = ya - np.mean(ya) # center y axis at zero
    y_mesh, x_mesh = np.meshgrid(ya, xa)

    # Blur function.
    R = params['blur_radius']
    circ = (x_mesh ** 2 + y_mesh **2 < R ** 2).astype(float)
    decay = (R - np.sqrt(x_mesh ** 2 + y_mesh ** 2)).astype(float)
    weight = circ * decay
    weight = torch.from_numpy(weight)[None, None, None, None, :, :]
    weight = weight.to(device) / torch.sum(weight)

    # Blur the unit cell permittivity.
    density = (eps_r - params['eps_min']) / (params['eps_max'] - params['eps_min'])
    density_blurred = convolve_density_with_blur(density, weight)

    return density_blurred * (params['eps_max'] - params['eps_min']) + params['eps_min']

def threshold(eps_r, params):
    '''
    This function applies a non-differentiable threshold operation to snap a 
    design to binary permittivity values.
    Args:
        eps_r: A `torch.Tensor` of shape `(1, pixelsX, pixelsY, Nlayer - 1, Nx, Ny)`
        and dtype `torch.float32` specifying the permittivity at each point in the 
        unit cell grid.
        params: A `dict` containing simulation and optimization settings.
    Returns:
        A `torch.Tensor` of dtype `torch.float32` and shape `(1, pixelsX, pixelsY, 
        Nlayers - 1, Nx, Ny)` specifying the binarized permittivity.
    '''
    # Apply the threshold.
    eps_thresh = eps_r.cpu().numpy()
    eps_thresh = (eps_thresh - params['eps_min']) / (params['eps_max'] - params['eps_min'])
    eps_thresh = eps_thresh > 0.5
    eps_thresh = eps_thresh * (params['eps_max'] - params['eps_min']) + params['eps_min']

    return torch.from_numpy(eps_thresh).float().to(eps_r.device)
