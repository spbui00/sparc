from sparc import MethodTester, AverageTemplateSubtraction, DataHandler
import numpy as np


def main():
    data_handler = DataHandler()
    data = data_handler.load_simulated_data('../research/generate_dataset/SimulatedData_2.mat')
    
    if data.artifact_indices is not None:
        print(f"\nLoaded data with {len(data.artifact_indices)} known artifact locations")
        print(f"First 10 artifact indices: {data.artifact_indices[:10]}")
        
        # Calculate average interval between artifacts (for setting template length)
        if len(data.artifact_indices) > 1:
            intervals = np.diff(data.artifact_indices)
            avg_interval = np.mean(intervals)
            print(f"Average interval between artifacts: {avg_interval:.1f} samples")
            print(f"At 30kHz, this is {avg_interval/30:.2f} ms")
    else:
        print("\nLoaded data with no known artifact locations.")
    
    # Create methods with different configurations
    tester = MethodTester(
        data=data,
        methods={
            # Short template (2ms) - good for short artifacts
            "avg_template_2ms": AverageTemplateSubtraction(
                template_length_ms=2,
                num_templates_for_avg=5
            ),
            # Medium template (5ms) - standard configuration
            "avg_template_5ms": AverageTemplateSubtraction(
                template_length_ms=5,
                num_templates_for_avg=5
            ),
            # Long template (10ms) - for longer artifacts
            "avg_template_10ms": AverageTemplateSubtraction(
                template_length_ms=10,
                num_templates_for_avg=3
            ),
        },
    )

    # Run the methods - they will automatically use the artifact_indices if available
    print("\n" + "=" * 50)
    print("Running artifact correction methods...")
    print("=" * 50)
    tester.run()
    
    # Print the results
    print("\n" + "=" * 50)
    print("Results Summary")
    print("=" * 50)
    tester.print_results()
    
    # Plot the results for visual inspection
    print("\n" + "=" * 50)
    print("Generating plots...")
    print("=" * 50)
    tester.plot_results(channel=0)


def compare_with_and_without_indices():
    """
    Compare performance when using known indices vs automatic detection.
    This is useful for validating that the known indices improve performance.
    """
    print("\n" + "=" * 50)
    print("Comparing: Known Indices vs Automatic Detection")
    print("=" * 50)
    
    data_handler = DataHandler()
    data = data_handler.load_simulated_data('../research/generate_dataset/SimulatedData_2.mat')
    
    # Method with known indices
    method_with_indices = AverageTemplateSubtraction(
        template_length_ms=5,
        num_templates_for_avg=5
    )
    
    # Method without indices (will detect automatically)
    method_without_indices = AverageTemplateSubtraction(
        template_length_ms=5,
        num_templates_for_avg=5
    )
    
    # Fit with indices
    print("\nFitting with known artifact indices...")
    method_with_indices.set_sampling_rate(data.sampling_rate)
    method_with_indices.fit(data.raw_data, artifact_indices=data.artifact_indices)
    cleaned_with_indices = method_with_indices.transform(data.raw_data)
    
    # Fit without indices (force automatic detection)
    print("Fitting with automatic artifact detection...")
    method_without_indices.set_sampling_rate(data.sampling_rate)
    method_without_indices.fit(data.raw_data, artifact_indices=None)  # Explicitly pass None
    cleaned_without_indices = method_without_indices.transform(data.raw_data)
    
    # Compare the results
    from sparc.core.evaluator import Evaluator
    evaluator = Evaluator(sampling_rate=data.sampling_rate)
    
    print("\n--- Results with Known Indices ---")
    results_with = evaluator.evaluate(
        ground_truth=data.ground_truth,
        original_mixed=data.raw_data,
        cleaned=cleaned_with_indices
    )
    
    print("\n--- Results with Automatic Detection ---")
    results_without = evaluator.evaluate(
        ground_truth=data.ground_truth,
        original_mixed=data.raw_data,
        cleaned=cleaned_without_indices
    )
    
    # Print comparison
    print("\n" + "=" * 50)
    print("Performance Comparison:")
    print("=" * 50)
    for metric in ['snr_improvement_db', 'correlation_coefficient', 'rmse']:
        if metric in results_with and metric in results_without:
            diff = results_with[metric] - results_without[metric]
            better = "Known" if diff > 0 else "Auto"
            print(f"{metric}:")
            print(f"  Known Indices:  {results_with[metric]:.4f}")
            print(f"  Auto Detection: {results_without[metric]:.4f}")
            print(f"  Difference:     {diff:+.4f} ({better} is better)")


if __name__ == "__main__":
    # Run the main demonstration
    main()
    
    # Optionally, run the comparison
    print("\n" + "=" * 70)
    print("Do you want to compare with automatic detection? (y/n): ", end="")
    try:
        user_input = input().strip().lower()
        if user_input == 'y':
            compare_with_and_without_indices()
    except:
        # In case of non-interactive environment
        print("Skipping comparison (non-interactive environment)")
