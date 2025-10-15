import numpy as np
from sklearn.linear_model import LinearRegression
from sparc import DataHandler
from sparc.core.signal_data import SimulatedData, ArtifactTriggers, SignalDataWithGroundTruth

def test_linear_mixing(data_obj, dataset_name):
    ground_truth = data_obj.ground_truth
    artifacts = data_obj.artifacts
    mixed_data = getattr(data_obj, 'raw_data', getattr(data_obj, 'mixed_data', None))
    
    if mixed_data is None:
        print(f"Error: Could not find mixed data for {dataset_name}")
        return 0

    n_trials, n_channels, n_time = mixed_data.shape

    r2_scores = []
    for i in range(n_channels):
        y_target = np.ravel(mixed_data[:, i, :])
        neural_source_ch = np.ravel(ground_truth[:, i, :])
        artifact_source_ch = np.ravel(artifacts[:, i, :])
        
        X_predictors = np.vstack([neural_source_ch, artifact_source_ch]).T
        model = LinearRegression()
        model.fit(X_predictors, y_target)
        score = model.score(X_predictors, y_target)
        r2_scores.append(score)

    if not r2_scores:
        print("Could not compute R-squared for any channel.")
        return 0.0

    avg_r2 = np.mean(r2_scores)
    print(f"Average R-squared across all channels: {avg_r2:.6f}")
    if avg_r2 > 0.95:
        print(f"[{dataset_name}] Result: The linear mixing assumption holds very strongly with an average R-squared of {avg_r2:.6f}.")
    else:
        print(f"[{dataset_name}] Result: The linear mixing assumption may be violated (significant non-linearities present) with an average R-squared of {avg_r2:.6f}.")
    print("-" * 40)
    return avg_r2

if __name__ == "__main__":
    data_handler = DataHandler()

    # --- Simulated Data 1000Hz ---
    data_obj = data_handler.load_npz_data('../../data/simulated_data_2x64_1000.npz')
    data_obj = SimulatedData(
        raw_data=data_obj['raw_data'], sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'], artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj['artifact_markers']),  
        firing_rate=data_obj['firing_rate'], spike_train=data_obj['spike_train'],
        lfp=data_obj['lfp'], stim_params=None, snr=data_obj['snr']
    )
    test_linear_mixing(data_obj, 'simulated data 1000Hz')

    # --- Simulated Data 30000Hz ---
    data_obj = data_handler.load_npz_data('../../data/simulated_data_2x64_30000.npz')
    data_obj = SimulatedData(
        raw_data=data_obj['raw_data'], sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'], artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj['artifact_markers']),  
        firing_rate=data_obj['firing_rate'], spike_train=data_obj['spike_train'],
        lfp=data_obj['lfp'], stim_params=None, snr=data_obj['snr']
    )
    test_linear_mixing(data_obj, 'simulated data 30000Hz')

    # --- SWEC Data ---
    data_obj = data_handler.load_npz_data('../../data/added_artifacts_swec_data_512.npz')
    data_obj = SignalDataWithGroundTruth(
        raw_data=data_obj['mixed_data'], sampling_rate=data_obj['sampling_rate'],
        ground_truth=data_obj['ground_truth'], artifacts=data_obj['artifacts'],
        artifact_markers=ArtifactTriggers(starts=data_obj['artifact_markers'])
    )
    test_linear_mixing(data_obj, 'swec data')

