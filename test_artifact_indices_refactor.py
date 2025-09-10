#!/usr/bin/env python3
"""
Test script to verify the artifact indices refactoring works correctly.
This script tests that:
1. Artifact indices are loaded from MATLAB files
2. They are passed correctly to the methods
3. The methods use them instead of detecting artifacts
"""

import numpy as np
import sys
import os

# Add the sparc module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sparc import DataHandler, AverageTemplateSubtraction


def test_loading_artifact_indices():
    """Test that artifact indices are loaded from MATLAB files."""
    print("=" * 60)
    print("TEST 1: Loading artifact indices from MATLAB file")
    print("=" * 60)
    
    data_handler = DataHandler()
    
    # Try to load simulated data
    try:
        data = data_handler.load_simulated_data('./research/generate_dataset/SimulatedData_2.mat')
        
        # Check if artifact indices were loaded
        if data.artifact_indices is not None:
            print(f"✓ Successfully loaded {len(data.artifact_indices)} artifact indices")
            print(f"  First 5 indices: {data.artifact_indices[:5]}")
            print(f"  Data shape: {data.raw_data.shape}")
            print(f"  Sampling rate: {data.sampling_rate} Hz")
            return True
        else:
            print("✗ No artifact indices found in the data")
            print("  (This might be expected if AllStimIdx is not in the .mat file)")
            return False
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False


def test_method_with_indices():
    """Test that methods can use provided artifact indices."""
    print("\n" + "=" * 60)
    print("TEST 2: Using artifact indices in method")
    print("=" * 60)
    
    # Create synthetic data and indices for testing
    np.random.seed(42)
    sampling_rate = 30000  # 30 kHz
    duration = 1  # 1 second
    n_samples = sampling_rate * duration
    n_channels = 2
    
    # Create synthetic signal
    data = np.random.randn(n_samples, n_channels) * 0.1
    
    # Create known artifact locations (every 100ms = 3000 samples)
    artifact_indices = np.arange(1000, n_samples, 3000)
    print(f"Created {len(artifact_indices)} synthetic artifact indices")
    
    # Add artifacts at known locations
    for idx in artifact_indices:
        if idx + 100 < n_samples:
            data[idx:idx+100, :] += np.random.randn(100, n_channels) * 2
    
    # Test method with indices
    method = AverageTemplateSubtraction(
        sampling_rate=sampling_rate,
        template_length_ms=5,
        num_templates_for_avg=3
    )
    
    # Fit with known indices
    print("Fitting method with known artifact indices...")
    method.fit(data, artifact_indices=artifact_indices)
    
    # Check that indices were stored
    if method.template_indices_ is not None:
        if isinstance(method.template_indices_, np.ndarray):
            print(f"✓ Method stored artifact indices: {len(method.template_indices_)} indices")
        else:
            print(f"✓ Method stored artifact indices")
        
        # Transform the data
        cleaned = method.transform(data)
        print(f"✓ Successfully transformed data")
        print(f"  Original data variance: {np.var(data):.4f}")
        print(f"  Cleaned data variance: {np.var(cleaned):.4f}")
        
        # Check that artifacts were reduced
        artifact_reduction = []
        for idx in artifact_indices[:3]:  # Check first 3 artifacts
            if idx + 100 < n_samples:
                orig_power = np.mean(data[idx:idx+100, 0]**2)
                clean_power = np.mean(cleaned[idx:idx+100, 0]**2)
                reduction = (orig_power - clean_power) / orig_power * 100
                artifact_reduction.append(reduction)
                print(f"  Artifact at index {idx}: {reduction:.1f}% power reduction")
        
        if np.mean(artifact_reduction) > 0:
            print(f"✓ Artifacts were reduced by average of {np.mean(artifact_reduction):.1f}%")
            return True
        else:
            print("✗ Artifacts were not reduced")
            return False
    else:
        print("✗ Method did not store artifact indices")
        return False


def test_comparison_with_without_indices():
    """Test that using known indices is different from automatic detection."""
    print("\n" + "=" * 60)
    print("TEST 3: Comparing known indices vs automatic detection")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    sampling_rate = 30000
    n_samples = 30000  # 1 second
    n_channels = 1
    
    data = np.random.randn(n_samples, n_channels) * 0.1
    
    # Add artifacts at irregular intervals (not evenly spaced)
    artifact_indices = np.array([1000, 4500, 8000, 11000, 15000, 19000, 23000, 27000])
    print(f"Created {len(artifact_indices)} artifacts at irregular intervals")
    
    for idx in artifact_indices:
        if idx + 150 < n_samples:
            data[idx:idx+150, :] += np.sin(np.linspace(0, 4*np.pi, 150)).reshape(-1, 1) * 2
    
    # Method 1: With known indices
    method_with = AverageTemplateSubtraction(
        sampling_rate=sampling_rate,
        template_length_ms=5,
        num_templates_for_avg=3
    )
    method_with.fit(data, artifact_indices=artifact_indices)
    cleaned_with = method_with.transform(data)
    
    # Method 2: Without indices (automatic detection)
    method_without = AverageTemplateSubtraction(
        sampling_rate=sampling_rate,
        template_length_ms=5,
        num_templates_for_avg=3,
        onset_threshold=2.0  # Set threshold for detection
    )
    method_without.fit(data, artifact_indices=None)
    cleaned_without = method_without.transform(data)
    
    # Compare results
    diff = np.mean(np.abs(cleaned_with - cleaned_without))
    print(f"Mean absolute difference between methods: {diff:.6f}")
    
    if diff > 0.001:  # They should be different
        print("✓ Methods produce different results (as expected)")
        print("  This confirms that known indices are being used differently")
        
        # Check which performs better at artifact locations
        artifact_power_with = []
        artifact_power_without = []
        for idx in artifact_indices[:3]:
            if idx + 150 < n_samples:
                power_with = np.mean(cleaned_with[idx:idx+150, 0]**2)
                power_without = np.mean(cleaned_without[idx:idx+150, 0]**2)
                artifact_power_with.append(power_with)
                artifact_power_without.append(power_without)
        
        avg_with = np.mean(artifact_power_with)
        avg_without = np.mean(artifact_power_without)
        print(f"  Avg artifact power (with indices): {avg_with:.6f}")
        print(f"  Avg artifact power (without):      {avg_without:.6f}")
        
        if avg_with < avg_without:
            print("✓ Known indices method performs better at artifact locations")
        return True
    else:
        print("✗ Methods produce identical results (unexpected)")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ARTIFACT INDICES REFACTORING TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Loading from MATLAB
    results.append(("Loading artifact indices", test_loading_artifact_indices()))
    
    # Test 2: Using indices in method
    results.append(("Using indices in method", test_method_with_indices()))
    
    # Test 3: Comparison
    results.append(("Comparison with/without indices", test_comparison_with_without_indices()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The refactoring is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
