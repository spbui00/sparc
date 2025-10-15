import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from scipy import signal
from sparc.core.data_handler import DataHandler
from sparc.core.signal_data import SignalDataWithGroundTruth, ArtifactTriggers
from sparc.core.plotting import NeuralAnalyzer
import matplotlib.pyplot as plt


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    idx = (freqs >= lo) & (freqs <= hi)
    return np.trapz(psd[:, idx], freqs[idx], axis=1)


def channel_tv_snapshot(data_tc: np.ndarray) -> float:
    diffs = np.abs(np.diff(data_tc, axis=0))
    return float(np.mean(diffs))


def temporal_roughness_snapshot(data_tc: np.ndarray) -> float:
    diffs = np.abs(np.diff(data_tc, axis=1))
    return float(np.mean(diffs))


def adjacent_channel_coherence(trace_tc: np.ndarray, fs: float) -> float:
    C = trace_tc.shape[0]
    if C < 2:
        return 0.0
    vals = []
    for c in range(C - 1):
        f, coh = signal.coherence(trace_tc[c], trace_tc[c + 1], fs=fs)
        vals.append(np.mean(coh))
    return float(np.mean(vals)) if vals else 0.0


def main():
    data_handler = DataHandler()
    raw = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')

    artifact_markers_data = raw['artifact_markers']
    if hasattr(artifact_markers_data, 'starts'):
        artifact_markers = artifact_markers_data
    else:
        artifact_markers = ArtifactTriggers(starts=artifact_markers_data)

    data_obj = SignalDataWithGroundTruth(
        raw_data=raw['mixed_data'],
        sampling_rate=raw['sampling_rate'],
        ground_truth=raw['ground_truth'],
        artifacts=raw['artifacts'],
        artifact_markers=artifact_markers
    )

    fs = float(np.asarray(data_obj.sampling_rate).item())
    analyzer = NeuralAnalyzer(sampling_rate=fs)
    lfp = analyzer.extract_lfp(data_obj.ground_truth, cutoff_freq=200.0)

    freqs, psd = analyzer.compute_psd(lfp, nperseg=512)
    bp_delta = band_power(freqs, psd, (1, 4))
    bp_theta = band_power(freqs, psd, (4, 8))
    bp_alpha = band_power(freqs, psd, (8, 12))
    bp_beta = band_power(freqs, psd, (12, 30))
    bp_gamma = band_power(freqs, psd, (30, 100))

    trial0 = lfp[0]
    ch_tv = channel_tv_snapshot(trial0)
    t_rough = temporal_roughness_snapshot(trial0)
    adj_coh = adjacent_channel_coherence(trial0, fs)

    print("SWEC LFP metrics (trial 0):")
    print(f"  channel-TV mean: {ch_tv:.4e}")
    print(f"  temporal roughness mean: {t_rough:.4e}")
    print(f"  adjacent-channel coherence: {adj_coh:.4f}")
    print("Band powers (mean across channels):")
    print(f"  delta 1-4 Hz : {np.mean(bp_delta):.4e}")
    print(f"  theta 4-8 Hz: {np.mean(bp_theta):.4e}")
    print(f"  alpha 8-12 Hz: {np.mean(bp_alpha):.4e}")
    print(f"  beta 12-30 Hz: {np.mean(bp_beta):.4e}")
    print(f"  gamma 30-100 Hz: {np.mean(bp_gamma):.4e}")

    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd.T)
    plt.title('SWEC LFP PSD (mean over trials)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True, linestyle='--', alpha=0.5)
    outdir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'swec_lfp_psd.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print(f"Saved PSD plot to {outpath}")


if __name__ == '__main__':
    main()


