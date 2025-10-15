import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sparc.core.data_handler import DataHandler
from sparc.core.signal_data import SimulatedData, ArtifactTriggers
from sparc.methods.decomposition.low_rank_tv import LowRankTV
from sparc.core.plotting import NeuralAnalyzer, NeuralPlotter


def low_rank_tv():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/simulated_data_2x64_30000.npz')

    data_obj = SimulatedData(
        raw_data=data_obj['raw_data'],
        sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'],
        artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj['artifact_markers']),
        firing_rate=data_obj['firing_rate'],
        spike_train=data_obj['spike_train'],
        lfp=data_obj['lfp'],
        stim_params=None,
        snr=data_obj['snr'],
    )

    print(f"Data shape: {data_obj.raw_data.shape}")
    print(f"Sampling rate: {data_obj.sampling_rate} Hz")

    method = LowRankTV(
        lambda_tv=0.1,
        rho=1.0,
        max_iters=400,
        tol=1e-4,
        features_axis=1,
        time_axis=2,
    )

    method.set_sampling_rate(data_obj.sampling_rate)

    print("Fitting LowRankTV...")
    method.fit(data_obj.raw_data)

    print("Transforming data...")
    cleaned_data = method.transform(data_obj.raw_data)

    # print min and max of cleaned data
    print(f"Min of cleaned data: {np.min(cleaned_data)}")
    print(f"Max of cleaned data: {np.max(cleaned_data)}")

    analyzer = NeuralAnalyzer(sampling_rate=data_obj.sampling_rate)
    plotter = NeuralPlotter(analyzer)

    plotter.plot_cleaned_comparison(
        data_obj.ground_truth,
        data_obj.raw_data,
        cleaned_data,
        0,
        1
    )

    plotter.plot_cleaned_comparison(
        data_obj.ground_truth,
        data_obj.raw_data,
        cleaned_data,
        0,
        20
    )


if __name__ == "__main__":
    low_rank_tv()


