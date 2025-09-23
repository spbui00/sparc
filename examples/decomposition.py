from sparc import MethodTester, DataHandler
from sparc.methods import ERAASR
import time


def main():
    data_handler = DataHandler()
    data = data_handler.load_concatenated_simulated_data('../data/SimulatedData_2x64_30000_10trials.npz', 30000)
    print(data.raw_data.shape)

    # for the stimulated data: samples_pre_train=0, n_pulses=8, samples_per_pulse=150
    fixed_params = {
        'samples_pre_train': 0,
        'samples_per_pulse': 150, # number of samples per pulse
        'n_pulses': 800 # number of pulses in the stimulus
    }

    grid_params = {
        'n_pc_channels': [4, 12],            # Number of PCs for channel cleaning
        'n_pc_pulses': [2, 6],               # Number of PCs for pulse cleaning
        'n_pc_trials': [2, 4],               # Number of PCs for trial cleaning
        'omit_bandwidth_channels': [1, 3],   # How many adjacent channels to ignore
        'pca_only_omitted': [True, False]    # Two different PCA model construction strategies
    }

    # only test one of the combinations
    grid_params['n_pc_channels'] = [4]
    grid_params['n_pc_pulses'] = [2]
    grid_params['n_pc_trials'] = [2]
    grid_params['omit_bandwidth_channels'] = [1]
    grid_params['pca_only_omitted'] = [True]

    methods = {}
    
    for n_ch in grid_params['n_pc_channels']:
        for n_pu in grid_params['n_pc_pulses']:
            for n_tr in grid_params['n_pc_trials']:
                for omit_bw in grid_params['omit_bandwidth_channels']:
                    for pca_omitted in grid_params['pca_only_omitted']:
                        pca_mode_str = "omitted" if pca_omitted else "full"
                        method_name = (
                            f"eraasr_ch{n_ch}_pu{n_pu}_tr{n_tr}_"
                            f"omit{omit_bw}_pca_{pca_mode_str}"
                        )
                        
                        methods[method_name] = ERAASR(
                            **fixed_params,
                            
                            # Pass the current parameters from the grid search
                            n_pc_channels=n_ch,
                            n_pc_pulses=n_pu,
                            n_pc_trials=n_tr,
                            omit_bandwidth_channels=omit_bw,
                            pca_only_omitted=pca_omitted,
                            
                            # Keep other advanced parameters at their robust defaults
                            omit_bandwidth_pulses=1,
                            omit_bandwidth_trials=1,
                            clean_over_trials_individual_channels=True
                        )

    tester = MethodTester(
        data=data,
        methods=methods,
        save=True,
        save_folder="../data/results/simulated/"
    )

    start_time = time.time()
    tester.run()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    tester.print_results()
    tester.compare()
    tester.plot_results()


if __name__ == "__main__":
    main()
