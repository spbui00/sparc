import numpy as np
from typing import Optional
from sparc import BaseSACMethod
import matplotlib.pyplot as plt


class SparseLocalProjection(BaseSACMethod):
    def __init__(self, stim_channel: int = None, local_radius: int = 5,
                 epoch_pre: int = 50, epoch_post: int = 150,
                 l1_alpha: float = 0.01, features_axis: Optional[int] = -1, **kwargs):
        super().__init__(**kwargs)
        self.stim_channel = stim_channel
        self.local_radius = local_radius
        self.epoch_pre = epoch_pre
        self.epoch_post = epoch_post
        self.l1_alpha = l1_alpha
        self._features_axis = features_axis
        self.spatial_map_ = None
        self.local_channels_ = None
        self.stim_times_ = None
        self._data_shape_template = None

    def fit(self, 
            data: np.ndarray, 
            stim_times: np.ndarray,
            **kwargs) -> 'SparseLocalProjection':
        if self.stim_channel is None:
            raise ValueError("stim_channel must be specified.")

        data_moved = np.moveaxis(data, self._features_axis, -1)
        self._data_shape_template = data_moved.shape  # (trials, time, ch)
        n_trials, n_time, n_ch = self._data_shape_template
        stim_times = np.asarray(stim_times)
        if stim_times.shape[0] != n_trials:
            raise ValueError("stim_times must have shape (n_trials, n_stims)")

        if self.stim_channel >= n_ch:
            raise ValueError(f"stim_channel out of range.")

        # Local channels
        local_start = max(0, self.stim_channel - self.local_radius)
        local_end = min(n_ch, self.stim_channel + self.local_radius + 1)
        self.local_channels_ = np.arange(local_start, local_end)
        n_local = len(self.local_channels_)

        # Initialize map
        all_a_local = []

        for trial in range(n_trials):
            trial_stim_times = stim_times[trial].flatten()
            for st in trial_stim_times:
                st = np.asarray(st).item()
                if st - self.epoch_pre < 0 or st + self.epoch_post > n_time:
                    continue  # Skip edge epochs
                epoch_slice = slice(st - self.epoch_pre, st + self.epoch_post)
                epoch_data = data_moved[trial, epoch_slice, :]  # (t_epoch, ch)
                local_epoch = epoch_data[:, self.local_channels_]  # (t_epoch, n_local)

                # Estimate s: mean over local channels
                s = np.mean(local_epoch, axis=1)  # (t_epoch,)
                ss = np.dot(s, s)
                if ss == 0:
                    continue

                # For each local channel, compute soft-thresholded a_j
                a_local = np.zeros(n_local)
                for j in range(n_local):
                    y = local_epoch[:, j]
                    ols_a = np.dot(s, y) / ss
                    shrink = self.l1_alpha / ss
                    a_local[j] = np.sign(ols_a) * max(0, abs(ols_a) - shrink)

                all_a_local.append(a_local)

        if not all_a_local:
            raise ValueError("No valid epochs found.")

        # Average over epochs
        mean_a_local = np.mean(all_a_local, axis=0)
        self.spatial_map_ = np.zeros(n_ch)
        self.spatial_map_[self.local_channels_] = mean_a_local

        self.stim_times_ = stim_times
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray, stim_times: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform.")
        if stim_times is None:
            stim_times = self.stim_times_
        stim_times = np.asarray(stim_times)

        data_moved = np.moveaxis(data, self._features_axis, -1)
        target_shape = data_moved.shape
        n_trials, n_time, n_ch = target_shape

        cleaned_data_moved = data_moved.copy()

        norm_a2 = np.dot(self.spatial_map_, self.spatial_map_)
        if norm_a2 == 0:
            return np.moveaxis(cleaned_data_moved, -1, self._features_axis)

        for trial in range(n_trials):
            trial_stim_times = stim_times[trial].flatten()
            for st in trial_stim_times:
                st = np.asarray(st).item()
                if st - self.epoch_pre < 0 or st + self.epoch_post > n_time:
                    continue
                epoch_slice = slice(st - self.epoch_pre, st + self.epoch_post)
                epoch_data = data_moved[trial, epoch_slice, :]  # (t_epoch, ch)
                # Compute proj_s = (a @ epoch_data.T) / ||a||^2
                # epoch_data.T is ch x t_epoch
                proj_s = np.dot(self.spatial_map_, epoch_data.T) / norm_a2  # (t_epoch,)
                # Artifact = spatial_map_[:, np.newaxis] * proj_s[np.newaxis, :]
                artifact = self.spatial_map_[:, np.newaxis] * proj_s  # ch x t
                # Subtract
                cleaned_epoch = epoch_data.T - artifact  # ch x t - ch x t = ch x t
                cleaned_data_moved[trial, epoch_slice, :] = cleaned_epoch.T  # t x ch

        # Note: Overlaps are overwritten; assume no overlap or handle if needed
        cleaned_data = np.moveaxis(cleaned_data_moved, -1, self._features_axis)
        return cleaned_data

    def plot_components(self, n_samples_to_plot: int = 1000):
        if not self.is_fitted:
            raise ValueError("Must fit before plotting.")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.spatial_map_)
        ax.set_title('Estimated Artifact Spatial Map')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Weight')
        ax.axvline(x=self.stim_channel, color='red', linestyle='--', label='Stim Channel')
        ax.legend()
        ax.grid(True, alpha=0.6)
        plt.tight_layout()
        plt.show()
