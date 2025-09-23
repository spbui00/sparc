import numpy as np
from typing import Optional
from sparc import BaseSACMethod
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


class ICA(BaseSACMethod):
    def __init__(self, n_components: int = 2, features_axis: Optional[int] = -1, **kwargs):
        super().__init__(**kwargs)
        self._features_axis = features_axis
        self.n_components = n_components
        self.ica_ = None
        self.S_ = None
        self.artifact_idx_ = None
        self._original_shape_info = None

    def _identify_artifact_component_idx(self, sources: np.ndarray) -> int:
        variances = np.var(sources, axis=0)
        artifact_index = np.argmax(variances)
        return artifact_index

    def fit(self, 
            data: np.ndarray, 
            **kwargs) -> 'ICA':
        data_moved = np.moveaxis(data, self._features_axis, -1)
        self._original_shape_info = data_moved.shape
        num_features = self._original_shape_info[-1]
        reshaped_data = data_moved.reshape(-1, num_features)

        n_components = self.n_components if self.n_components is not None else num_features
        if n_components > num_features:
            raise ValueError(f"n_components ({n_components}) cannot be greater than the number of features ({num_features}).")

        self.ica_ = FastICA(n_components=n_components, **kwargs)
        self.S_ = self.ica_.fit_transform(reshaped_data)
        self.artifact_idx_ = self._identify_artifact_component_idx(self.S_)

        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Method must be fitted before transforming data.")

        data_moved = np.moveaxis(data, self._features_axis, -1)
        num_features = data_moved.shape[-1]
        reshaped_data = data_moved.reshape(-1, num_features)
        
        sources = self.ica_.transform(reshaped_data)
        sources[:, self.artifact_idx_] = 0
        cleaned_reshaped_data = self.ica_.inverse_transform(sources)

        cleaned_data_moved = cleaned_reshaped_data.reshape(self._original_shape_info)
        cleaned_data = np.moveaxis(cleaned_data_moved, -1, self._features_axis)

        return cleaned_data

    def plot_components(self, n_samples_to_plot: int = 1000):
        if not self.is_fitted:
            raise ValueError("Method must be fitted before plotting components.")
            
        n_components = self.S_.shape[1]
        plot_data = self.S_[:n_samples_to_plot, :]
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 2 * n_components), sharex=True)
        if n_components == 1: axes = [axes]
        fig.suptitle('Independent Components (Sources)', fontsize=16)
        
        for i, ax in enumerate(axes):
            ax.plot(plot_data[:, i])
            title = f'Component {i}'
            if i == self.artifact_idx_:
                title += ' (Identified as Artifact)'
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.6)
            
        axes[-1].set_xlabel('Time (samples)')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def main():
    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    s1 = np.sin(2 * np.pi * 5 * time)
    s2 = np.zeros_like(time)
    s2[::200] = 5.0
    s2[1::200] = -5.0
    S = np.c_[s1, s2]
    A = np.array([[1, 1], [0.5, 2], [1.5, 0.5]])
    X_mixed = S.dot(A.T)
    X_3d = X_mixed.T.reshape(1, 3, n_samples)

    ica_model = ICA(n_components=2, features_axis=1)
    ica_model.fit(X_3d, random_state=0, max_iter=500)
    ica_model.plot_components(n_samples_to_plot=2000)
    cleaned_data = ica_model.transform(X_3d)

    channel_to_plot = 1
    ground_truth_neural = A[channel_to_plot, 0] * s1
    plt.figure(figsize=(12, 6))
    plt.plot(time, X_3d[0, channel_to_plot, :], label='Original Mixed Signal', color='orange', alpha=0.7)
    plt.plot(time, ground_truth_neural, label='Ground Truth Neural Signal', color='gray', linestyle='--', linewidth=2)
    plt.plot(time, cleaned_data[0, channel_to_plot, :], label='Cleaned Signal (ICA Output)', color='blue', alpha=0.8)
    plt.title(f'Comparison on Channel {channel_to_plot} (Full View)', fontsize=16)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()