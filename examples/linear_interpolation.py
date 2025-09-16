from sparc.methods import LinearInterpolation
from sparc import DataHandler, MethodTester


def main():
    data_handler = DataHandler()
    data = data_handler.load_simulated_data('../research/generate_dataset/SimulatedData_2.mat', sampling_rate=30000)

    window_sizes = [0.0001]
    margins = [0.1, 0.2, 0.3]

    methods = {}

    for ws in window_sizes:
        for mg in margins:
            method_name = f"linear_interpolation_{ws}_{mg}"
            method = LinearInterpolation(window_size_ms=ws, margin_ms=mg)
            methods[method_name] = method
    
    tester = MethodTester(
        data=data,
        methods=methods,
    )

    tester.run()
    tester.print_results()
    tester.plot_results()
    tester.compare()


if __name__ == "__main__":
    main()
