from sparc import MethodTester, DataHandler
from sparc.methods import ERAASR

def main():
    data_handler = DataHandler()
    data = data_handler.load_simulated_data('../research/generate_dataset/SimulatedData_1_1024.mat', sampling_rate=30000)
    print(f"number of channels: {data.raw_data.shape[1]}")
    
    n_pc_channels_options = [4, 8, 12]
    n_pc_pulses_options = [2, 4, 6]
    n_pc_trials_options = [1, 2, 4]

    methods = {}

    for n_ch in n_pc_channels_options:
        for n_pu in n_pc_pulses_options:
            for n_tr in n_pc_trials_options:
                method_name = f"eraasr_ch{n_ch}_pu{n_pu}_tr{n_tr}"
                method = ERAASR(
                    n_pc_channels=n_ch,
                    n_pc_pulses=n_pu,
                    n_pc_trials=n_tr,
                    samples_per_pulse=150,
                    n_pulses=8,
                )
                methods[method_name] = method
    
    tester = MethodTester(
        data=data,
        methods=methods,
    )

    eraasr_data = data_handler.load_eraasr_data('../research/datasets/eraasr-1.0.0/exampleDataTensor.mat', sampling_rate=30000)

    tester.run()
    tester.print_results()
    tester.compare()
    # tester.plot_results()


if __name__ == "__main__":
    main()
