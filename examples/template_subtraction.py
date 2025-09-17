from sparc import MethodTester, DataHandler
from sparc.methods import BackwardTemplateSubtraction


def main():
    data_handler = DataHandler()
    data = data_handler.load_simulated_data('../research/generate_dataset/SimulatedData_1_1024.npz', sampling_rate=30000)
    
    tester = MethodTester(
        data=data,
        methods={
            "avg_template_subtraction_0.8_5": BackwardTemplateSubtraction(
                template_length_ms=0.8,
                num_templates_for_avg=5
            ),
            "avg_template_subtraction_0.9_6": BackwardTemplateSubtraction(
                template_length_ms=0.9,
                num_templates_for_avg=6
            ),
            "avg_template_subtraction_1.0_7": BackwardTemplateSubtraction(
                template_length_ms=1.0,
                num_templates_for_avg=5
            ),
            "avg_template_subtraction_1.0_7": BackwardTemplateSubtraction(
                template_length_ms=1.0,
                num_templates_for_avg=7
            ),
        },
    )

    tester.run()
    tester.print_results()
    tester.compare()
    tester.plot_results()


if __name__ == "__main__":
    main()
