from sparc import MethodTester, DataHandler, NeuralAnalyzer, NeuralPlotter
from sparc.methods import BackwardTemplateSubtraction


def main():
    data_handler = DataHandler()
    data = data_handler.load_simulated_data('../data/simulated_data_2x64_30000.npz', sampling_rate=30000)
    
    template_lengths = [0.03, 0.05, 0.5, 0.8]
    num_templates = [1,3,5]

    methods = {}
    for tl in template_lengths:
        for nt in num_templates:
            method_name = f"avg_template_subtraction_{tl}_{nt}"
            methods[method_name] = BackwardTemplateSubtraction(
                template_length_ms=tl,
                num_templates_for_avg=nt
            )

    
    tester = MethodTester(
        data=data,
        methods=methods,
    )

    tester.run()
    tester.print_results()
    tester.compare()
    tester.plot_results()


if __name__ == "__main__":
    main()
