import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt
from typing import Tuple


class PhysicsLoss(nn.Module):
    def __init__(self, 
                w_cosine=1.0, 
                w_rank=1.0, 
                w_smooth=1.0, 
                w_spectral=1.0, 
                w_wavelet_energy=1.0, 
                w_wavelet_sparsity=1.0, 
                w_wavelet_entropy=1.0, 
                sampling_rate=1000, 
                f_cutoff=200.0, eps=1e-8):
        super(PhysicsLoss, self).__init__()
        self.w_cosine = w_cosine
        self.w_rank = w_rank
        self.w_smooth = w_smooth
        self.w_spectral = w_spectral
        self.wavelet = 'db4'
        self.levels = 5
        self.w_wavelet_energy = w_wavelet_energy
        self.w_wavelet_sparsity = w_wavelet_sparsity
        self.w_wavelet_entropy = w_wavelet_entropy
        self.sampling_rate = sampling_rate
        self.f_cutoff = f_cutoff
        self.eps = eps

    def _spectral_loss(self, s_pred: torch.Tensor) -> torch.Tensor:
        n_samples = s_pred.shape[-1]
        s_fft = torch.fft.rfft(s_pred, n=n_samples, dim=-1)
        s_psd = torch.abs(s_fft)**2
        freqs = torch.fft.rfftfreq(n=n_samples, d=1.0/self.sampling_rate, device=s_pred.device)
        cutoff_idx = torch.searchsorted(freqs, self.f_cutoff)
        p_total = torch.sum(s_psd, dim=-1)
        p_low = torch.sum(s_psd[..., :cutoff_idx], dim=-1)
        loss = 1.0 - (p_low / (p_total + self.eps))
        return torch.mean(loss)

    def _get_spectral_ratio(self, x: torch.Tensor) -> torch.Tensor:
        # This function returns the ratio of high-frequency power
        n_samples = x.shape[-1]
        s_fft = torch.fft.rfft(x, n=n_samples, dim=-1)
        s_psd = torch.abs(s_fft)**2
        freqs = torch.fft.rfftfreq(n=n_samples, d=1.0/self.sampling_rate, device=x.device)
        
        cutoff_idx = torch.searchsorted(freqs, self.f_cutoff)
        p_total = torch.sum(s_psd, dim=-1) + self.eps
        p_low = torch.sum(s_psd[..., :cutoff_idx], dim=-1)
        
        # Ratio of high-frequency power
        p_high_ratio = (p_total - p_low) / p_total 
        return torch.mean(p_high_ratio)

    def _cov(self, M: torch.Tensor) -> torch.Tensor:
        T, C, N = M.shape
        M_mean = torch.mean(M, dim=-1, keepdim=True)
        M_centered = M - M_mean
        
        Cov = (M_centered @ M_centered.transpose(-1, -2)) / (N - 1)
        return Cov

    def _logdet(self, Cov: torch.Tensor) -> torch.Tensor:
        T, C, _ = Cov.shape
        eps_I = torch.eye(C, device=Cov.device, dtype=Cov.dtype).unsqueeze(0) * self.eps
        Cov_stable = Cov + eps_I
        
        _, logdet = torch.linalg.slogdet(Cov_stable)
        return logdet

    def _compute_wavelet_properties(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.zeros(x.shape[0], 3), torch.zeros(x.shape[0], 3), torch.zeros(x.shape[0], 3)
        batch, channels, time = x.shape
        energy_dists = []
        sparsities = []
        entropies = []
        
        for b in range(batch):
            for c in range(channels):
                signal = x[b, c].unsqueeze(0)
                
                coeffs = ptwt.wavedec(signal, self.wavelet, level=self.levels, mode='periodic')
                
                energies = torch.stack([torch.sum(coeff**2) for coeff in coeffs])
                total_energy = torch.sum(energies) + self.eps
                energy_dist = energies / total_energy
                energy_dists.append(energy_dist)
                
                all_coeffs = torch.cat([c.flatten() for c in coeffs])
                all_coeffs_abs = torch.abs(all_coeffs)
                sorted_coeffs, _ = torch.sort(all_coeffs_abs)
                n = sorted_coeffs.shape[0]
                indices = torch.arange(n, dtype=sorted_coeffs.dtype, device=sorted_coeffs.device)
                gini = 1 - 2 * torch.sum(sorted_coeffs * (n - indices + 0.5)) / (n * torch.sum(sorted_coeffs) + self.eps)
                sparsities.append(gini)
                
                energy_dist_normalized = energy_dist / (torch.sum(energy_dist) + self.eps)
                entropy = -torch.sum(energy_dist_normalized * torch.log(energy_dist_normalized + self.eps))
                entropies.append(entropy)
        
        return (torch.stack(energy_dists),
                torch.stack(sparsities),
                torch.stack(entropies))

    def forward(self, s_pred: torch.Tensor, a_pred: torch.Tensor) -> torch.Tensor:
        B = a_pred.shape[0]

        s_flat = s_pred.reshape(B, -1)
        a_flat = a_pred.reshape(B, -1)
        
        # cosine similarity between s_pred and a_pred
        cos_sim = F.cosine_similarity(s_flat, a_flat, dim=1, eps=self.eps)
        loss_cosine = torch.mean(cos_sim**2)

        # total variation loss: neural should be smoother than artifact (tv_s < tv_a)
        # Using ratio encourages relative smoothness without forcing neural to be flat
        tv_s = torch.mean(torch.abs(s_pred[..., 1:] - s_pred[..., :-1]))
        tv_a = torch.mean(torch.abs(a_pred[..., 1:] - a_pred[..., :-1]))
        loss_smooth = tv_s / (tv_a + self.eps)

        cov_s = self._cov(s_pred)
        cov_a = self._cov(a_pred)
        logdet_s = self._logdet(cov_s)
        logdet_a = self._logdet(cov_a)
        loss_rank = torch.mean(logdet_a - logdet_s) 

        loss_spectral_s = self._get_spectral_ratio(s_pred)
        loss_spectral_a = 1.0 - self._get_spectral_ratio(a_pred)
        loss_spectral = loss_spectral_s + loss_spectral_a

        s_energy, s_sparsity, s_entropy = self._compute_wavelet_properties(s_pred)
        a_energy, a_sparsity, a_entropy = self._compute_wavelet_properties(a_pred)

        neural_energy_loss = torch.mean((s_energy[:, 0] - 0.95)**2)  # Target 95% in level 0
        sparsity_loss = torch.relu(a_sparsity.mean() - s_sparsity.mean())
        entropy_loss = torch.relu(s_entropy.mean() - a_entropy.mean())

        total_loss = (
                      self.w_cosine * loss_cosine +
                      self.w_rank * loss_rank +
                      self.w_smooth * loss_smooth +
                      self.w_spectral * loss_spectral +
                      self.w_wavelet_energy * neural_energy_loss +
                      self.w_wavelet_sparsity * sparsity_loss +
                      self.w_wavelet_entropy * entropy_loss
                      )
        
        return total_loss
