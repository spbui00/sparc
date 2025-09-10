from sparc import MethodTester, AverageTemplateSubtraction, DataHandler


def main():
    print("Template Subtraction Demonstration on Simulated Data")
    print("=" * 50)

    data_handler = DataHandler()
    data = data_handler.load_simulated_data('../research/generate_dataset/SimulatedData_2.mat')
    
    tester = MethodTester(
        data=data,
        methods={
            "avg_template_subtraction": AverageTemplateSubtraction(
                template_length_ms=2,
                num_templates_for_avg=5
            ),
            "avg_template_subtraction_long": AverageTemplateSubtraction(
                template_length_ms=4,
                num_templates_for_avg=5
            ),
        },
    )

    tester.run()
    tester.print_results()
    tester.plot_results()


if __name__ == "__main__":
    main()
