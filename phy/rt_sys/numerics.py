"""Batched complex RZF/LMMSE, on NumPy for fixtures or PyTorch for production.

SINR uses instantaneous absolute channels, explicitly including every active
stream. No empirical beam, MU, shadowing or inter-cell penalty is added.
"""
import numpy as np


class Algebra:
    def __init__(self, device='numpy'):
        self.device = device
        self.torch = None
        if device != 'numpy':
            import torch
            self.torch = torch
            if device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable')

    def array(self, x):
        if self.torch:
            return self.torch.as_tensor(x, dtype=self.torch.complex64, device=self.device)
        return np.asarray(x, dtype=np.complex128)

    def numpy(self, x):
        return x.detach().cpu().numpy() if self.torch else np.asarray(x)

    def eye(self, n):
        return self.array(np.eye(n))

    def adj(self, x):
        return x.conj().swapaxes(-2, -1)

    def solve(self, a, b):
        return self.torch.linalg.solve(a, b) if self.torch else np.linalg.solve(a, b)

    def svd(self, x):
        return self.torch.linalg.svd(x, full_matrices=False) if self.torch else np.linalg.svd(x, full_matrices=False)

    def sum(self, x, axis, keepdims=False):
        return x.sum(dim=axis, keepdim=keepdims) if self.torch else x.sum(axis=axis, keepdims=keepdims)

    def clip(self, x, low):
        return x.clamp_min(low) if self.torch else np.maximum(x, low)

    def cat(self, x, axis=-1):
        return self.torch.cat(x, dim=axis) if self.torch else np.concatenate(x, axis=axis)

    def rzf(self, h, basis, alpha):
        # h [frequency, total streams, TX]; basis [TX, subspace]
        effective = h @ basis
        w = basis @ self.adj(effective) @ self.solve(
            effective @ self.adj(effective) + alpha * self.eye(effective.shape[-2]),
            self.eye(effective.shape[-2]))
        norm = self.clip(self.sum(abs(w)**2, -2, True), 1e-30)**.5
        return w / norm

    def lmmse(self, desired, others, noise):
        # Desired stream [frequency,RX,1], all other received streams [frequency,RX,K]
        covariance = others @ self.adj(others) + noise * self.eye(desired.shape[-2])
        sinr = (self.adj(desired) @ self.solve(covariance, desired))[..., 0, 0].real
        return self.clip(sinr, 0)


def beam_subspace(nt, codebook_id, profile_id, beam_id):
    """DFT subspaces. IDs specify physical weights, never scalar gain bonuses.

    Baseline uses all DFT directions; directional uses half, coverage uses 3/4.
    Narrow profiles reduce the subspace by one (minimum two for MU support).
    """
    if codebook_id not in (0, 1, 2) or profile_id not in (0, 1):
        raise ValueError(f'Unmapped physical codebook/profile: {codebook_id}/{profile_id}')
    sizes = [nt, max(1, nt//2), max(1, (3*nt)//4)]
    size = max(min(2, nt), sizes[codebook_id] - profile_id)
    indices = [(beam_id + j) % nt for j in range(size)]
    dft = np.exp(2j*np.pi*np.arange(nt)[:, None]*np.arange(nt)[None, :]/nt)/np.sqrt(nt)
    return dft[:, indices]
