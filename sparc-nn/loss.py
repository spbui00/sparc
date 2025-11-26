import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    def __init__(self, 
                 w_cosine=1.0, 
                 w_rank=1.0, 
                 w_smooth=1.0, 
                 w_spectral=1.0, 
                 sampling_rate=1000, 
                 f_cutoff=200.0, eps=1e-8):
        super(PhysicsLoss, self).__init__()
        self.w_cosine = w_cosine
        self.w_rank = w_rank
        self.w_smooth = w_smooth
        self.w_spectral = w_spectral
        self.sampling_rate = sampling_rate
        self.f_cutoff = f_cutoff
        self.eps = eps

    def _spectral_slope_loss(self, x: torch.Tensor) -> torch.Tensor:        
        s_fft = torch.fft.rfft(x, dim=-1)
        s_log_psd = torch.log(torch.abs(s_fft)**2 + self.eps)

        diff = s_log_psd[..., 1:] - s_log_psd[..., :-1]

        positive_slope_penalty = torch.relu(diff) ** 2
        return torch.mean(positive_slope_penalty)

    def _get_spectral_ratio(self, x: torch.Tensor) -> torch.Tensor:
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
        """Covariance between channels"""
        T, C, N = M.shape
        M_mean = torch.mean(M, dim=-1, keepdim=True)
        M_centered = M - M_mean

        Cov = (M_centered @ M_centered.transpose(-1, -2)) / (N - 1)
        return Cov

    def forward(self, s_pred: torch.Tensor, a_pred: torch.Tensor) -> dict:
        # cosine similarity between s_pred and a_pred
        cos_sim_per_channel = F.cosine_similarity(s_pred, a_pred, dim=2, eps=self.eps)
        loss_cosine = torch.mean(cos_sim_per_channel**2)

        cov_a = self._cov(a_pred)
        nuc_a = torch.linalg.norm(cov_a, ord='nuc', dim=(-2, -1))
        fro_a = torch.linalg.norm(cov_a, ord='fro', dim=(-2, -1))
        soft_rank_a = nuc_a / (fro_a + 1e-6)
        loss_rank_a = torch.mean(torch.abs(soft_rank_a - 1.0))

        cov_s = self._cov(s_pred)
        nuc_s = torch.linalg.norm(cov_s, ord='nuc', dim=(-2, -1))
        fro_s = torch.linalg.norm(cov_s, ord='fro', dim=(-2, -1))
        soft_rank_s = nuc_s / (fro_s + 1e-6)
        loss_rank_s_penalty = torch.mean(2 - (soft_rank_s))

        loss_spectral_s = self._get_spectral_ratio(s_pred)
        loss_spectral_a = 1.0 - self._get_spectral_ratio(a_pred)
        loss_spectral = loss_spectral_s + loss_spectral_a

        loss_spectral_slope_s = self._spectral_slope_loss(s_pred)

        return {
            'cosine': loss_cosine,
            'rank_a': loss_rank_a,
            'rank_s_penalty': loss_rank_s_penalty,
            'spectral': loss_spectral,
            'spectral_slope_s': loss_spectral_slope_s,
        }

        return total_loss
