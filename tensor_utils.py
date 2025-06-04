import torch
import numpy as np
from torch.autograd import Function

def expand_and_tile_np(array, batchSize, pixelsX, pixelsY):
  '''
    Expands and tile a numpy array for a given batchSize and number of pixels.
    Args:
        array: A `np.ndarray` of shape `(Nx, Ny)`.
    Returns:
        A `np.ndarray` of shape `(batchSize, pixelsX, pixelsY, Nx, Ny)` with
        the values from `array` tiled over the new dimensions.
  '''
  array = array[np.newaxis, np.newaxis, np.newaxis, :, :]
  return np.tile(array, reps = (batchSize, pixelsX, pixelsY, 1, 1))

def expand_and_tile_torch(tensor, batchSize, pixelsX, pixelsY):
  '''
    Expands and tile a `torch.Tensor` for a given batchSize and number of pixels.
    Args:
        tensor: A `torch.Tensor` of shape `(Nx, Ny)`.
    Returns:
        A `torch.Tensor` of shape `(batchSize, pixelsX, pixelsY, Nx, Ny)` with
        the values from `tensor` tiled over the new dimensions.
  '''
  tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
  return tensor.repeat(batchSize, pixelsX, pixelsY, 1, 1)

class EigGeneral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, eps=1E-6):
        A = A.to(torch.complex64).to("cuda")  # Ensure that A is of type complex64
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        
        ctx.save_for_backward(A, eigenvalues, eigenvectors)
        ctx.eps = eps
        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_D, grad_U):
        A, D, U = ctx.saved_tensors
        eps = ctx.eps
        batchSize, pixelsX, pixelsY, Nlay, dim, _ = A.shape

        # Convert eigenvalues gradient to a diagonal matrix.
        grad_D = torch.diag_embed(grad_D)

        # Calculate intermediate matrices.
        I = torch.eye(dim, dtype=A.dtype, device=A.device).to("cuda")
        D = D.view(batchSize, pixelsX, pixelsY, Nlay, dim, 1)
        E = D - D.transpose(-2, -1)

        # Lorentzian broadening.
        F = E / (E ** 2 + eps)
        F = F - I * F

        # Compute the reverse mode gradient of the eigendecomposition of A.
        grad_A = torch.conj(F).to("cuda") *  torch.matmul(U.transpose(-2, -1), grad_U).to("cuda")
        grad_A = grad_D + grad_A
        grad_A = torch.matmul(grad_A, U.transpose(-2, -1)).to("cuda")
        grad_A = torch.matmul(torch.linalg.inv(U.transpose(-2, -1)), grad_A).to("cuda")
        del A
        torch.cuda.empty_cache()
        return grad_A, None

# Usage:
# eigenvalues, eigenvectors = EigGeneral.apply(A)
