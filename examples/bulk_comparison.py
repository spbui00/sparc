from sparc import MethodTester, DataHandler
from sparc.methods import ERAASR, BackwardTemplateSubtraction, LinearInterpolation
import time
import os


def main():
    data_handler = DataHandler()
    data = data_handler.load_concatenated_simulated_data('../data/SimulatedData_2x64_30000_10trials.npz', 30000)
    print(data.raw_data.shape)

    save_folder = "../data/results/simulated/"

    all_methods = {
        **_template_subtraction(),
        **_linear_interpolation(),
        **_decomposition()
    }

    # Filter out methods that already have results
    methods = {}
    skipped_count = 0
    for name, method in all_methods.items():
        cleaned_file = os.path.join(save_folder, f"{name}_cleaned.npy")
        config_file = os.path.join(save_folder, f"{name}_config.json")
        
        if os.path.exists(cleaned_file) and os.path.exists(config_file):
            print(f"⏭️  Skipping {name} (already completed)")
            skipped_count += 1
        else:
            methods[name] = method
    
    print(f"Running {len(methods)}/{len(all_methods)} methods")

    tester = MethodTester(
        data=data,
        methods=methods,
        save=True,
        save_folder=save_folder
    )

    start_time = time.time()
    tester.run()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    tester.print_results()
    tester.compare()
    tester.plot_results()

def _decomposition():
    fixed_params = {
        'samples_pre_train': 0,
        'samples_per_pulse': 150, # number of samples per pulse
        'n_pulses': 800 # number of pulses in the stimulus
    }

    grid_params = {
        'n_pc_channels': [4, 10],            # Number of PCs for channel cleaning
        'n_pc_pulses': [2],               # Number of PCs for pulse cleaning
        'n_pc_trials': [2, 4],               # Number of PCs for trial cleaning
        'omit_bandwidth_channels': [4],   # How many adjacent channels to ignore
        'omit_bandwidth_pulses': [1],   # How many adjacent pulses to ignore
        'omit_bandwidth_trials': [0],   # How many adjacent trials to ignore
        'pca_only_omitted': [True]    # Two different PCA model construction strategies
    }

    methods = {}
    
    for n_ch in grid_params['n_pc_channels']:
        for n_pu in grid_params['n_pc_pulses']:
            for n_tr in grid_params['n_pc_trials']:
                for omit_bw in grid_params['omit_bandwidth_channels']:
                    for omit_bw_pu in grid_params['omit_bandwidth_pulses']:
                        for omit_bw_tr in grid_params['omit_bandwidth_trials']:
                            for pca_omitted in grid_params['pca_only_omitted']:
                                pca_mode_str = "omitted" if pca_omitted else "full"
                                method_name = (
                                    f"eraasr_ch{n_ch}_pu{n_pu}_tr{n_tr}_"
                                    f"omit{omit_bw}_pu{omit_bw_pu}_tr{omit_bw_tr}_pca_{pca_mode_str}"
                                )
                                
                                methods[method_name] = ERAASR(
                                    **fixed_params,
                                    n_pc_channels=n_ch,
                                    n_pc_pulses=n_pu,
                                    n_pc_trials=n_tr,
                                    omit_bandwidth_channels=omit_bw,
                                    omit_bandwidth_pulses=omit_bw_pu,
                                    omit_bandwidth_trials=omit_bw_tr,
                                    pca_only_omitted=pca_omitted,
                                    clean_over_trials_individual_channels=True
                                )
    return methods

def _template_subtraction():
    template_lengths = [1.0, 2.0, 3.0, 5.0]
    num_templates = [1,3,5]

    methods = {}
    for tl in template_lengths:
        for nt in num_templates:
            method_name = f"avg_template_subtraction_{tl}_{nt}"
            methods[method_name] = BackwardTemplateSubtraction(
                template_length_ms=tl,
                num_templates_for_avg=nt
            )
    return methods

def _linear_interpolation():
    artifact_durations = [1.0, 2.0, 3.0, 4.0]  # ms
    margins = [0.1, 0.2, 0.3]                   # ms

    methods = {}

    for ad in artifact_durations:
        for mg in margins:
            method_name = f"linear_interpolation_{ad}_{mg}"
            method = LinearInterpolation(artifact_duration_ms=ad, margin_ms=mg)  # Fix: artifact_duration_ms not window_size_ms
            methods[method_name] = method
    return methods

if __name__ == "__main__":
    main()
