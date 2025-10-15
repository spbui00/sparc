import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from sparc.core.data_handler import DataHandler
from sparc.core.signal_data import SignalData
from sparc.methods.decomposition.low_rank_tv import LowRankTV
from sparc.core.plotting import NeuralAnalyzer, NeuralPlotter


def low_rank_tv():
    data_handler = DataHandler()
    data_obj = data_handler.load_npz_data('../../data/eraasr_1000.npz')
    
    one_trial = data_obj['arr_0'][0:1]
    data_obj = SignalData(
        raw_data=one_trial,
        sampling_rate=1000,
    )
    print(f"Data shape: {data_obj.raw_data.shape}")
    print(f"Sampling rate: {data_obj.sampling_rate} Hz")

    method = LowRankTV(
        lambda_tv=1.0,
        rho=1.0,
        max_iters=1000,
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

    plotter.plot_trace_comparison(cleaned_data, data_obj.raw_data, 0, 0, "LowRankTV")
    plotter.plot_trace_comparison(cleaned_data, data_obj.raw_data, 0, 20, "LowRankTV")


if __name__ == "__main__":
    low_rank_tv()


