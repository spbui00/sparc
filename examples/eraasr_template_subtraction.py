import numpy as np
import matplotlib.pyplot as plt
from sparc import MethodTester, DataHandler
from sparc.core.signal_data import SignalDataWithGroundTruth
from sparc.methods import BackwardTemplateSubtraction
from sparc.methods.decomposition import eraasr


def main():
    data_handler = DataHandler() 
    sampling_rate = 300000
    eraasr_data = data_handler.load_eraasr_data('../research/datasets/eraasr-1.0.0/exampleDataTensor.mat', sampling_rate=sampling_rate)

    methods = {
        '2ms_template': BackwardTemplateSubtraction(
            template_length_ms=2,
            num_templates_for_avg=5
        ),
        '5ms_template': BackwardTemplateSubtraction(
            template_length_ms=5,
            num_templates_for_avg=5
        )
    }
    
    tester = MethodTester(
        data=eraasr_data,
        methods=methods,
    )
    
    # 3. Run the tests
    tester.run()
    
    # 4. Get and print results
    results = tester.get_results()
    
    print("\n=== Test Results ===")
    for method_name, result in results.items():
        print(f"\n--- {method_name} ---")
        print(f"  Overall Power Reduction: {result['power_reduction']:.2f}%")
        print(f"  LFP Power Reduction: {result['lfp_power_reduction']:.2f}%")
        print(f"  MUA Power Reduction: {result['mua_power_reduction']:.2f}%")
        if "spike_metrics" in result:
            print("  Spike Metrics (Channel 0):")
            metrics = result['spike_metrics']['channel_0']
            print(f"    Hits: {metrics['hits']}")
            print(f"    Misses: {metrics['misses']}")
            print(f"    False Positives: {metrics['false_positives']}")
            print(f"    Precision: {metrics['precision']:.2f}")
            print(f"    Recall: {metrics['recall']:.2f}")

    # 5. Plot the results
    plot_results(tester)

if __name__ == "__main__":
    main()
