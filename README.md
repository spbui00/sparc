# SPARC/README.md
# SPARC: Stimulation artifact Purging and Analysis in Real-time Circuits

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/sparc-neural.svg)](https://badge.fury.io/py/sparc-neural)

A comprehensive Python library for removing stimulation artifacts from neural recordings and analyzing neural signals in real-time.

## Features

### Artifact Removal Methods
- **Interpolation**: Linear/spline interpolation over artifact regions
- **Template Subtraction**: Learn and subtract artifact templates
- **SVD Decomposition**: Remove largest singular values
- **Dictionary Learning**: Advanced template matching (coming soon)

### Neural Signal Analysis
- **Spectral Analysis**: PSD, spectrograms, wavelets
- **Signal Quality**: SNR, artifact detection, coherence
- **Feature Extraction**: RMS, variance, skewness, kurtosis, spectral centroid
- **Real-time Processing**: Optimized for streaming data

### Evaluation & Visualization
- **Comprehensive Metrics**: MSE, PSD MSE, SNR improvement, spectral coherence
- **Method Comparison**: Side-by-side evaluation of different approaches
- **Advanced Plotting**: Time series, spectral, and feature analysis plots

## Installation

### From PyPI (when published)
```bash
pip install sparc-neural
```

### From source
```bash
git clone https://github.com/yourusername/sparc-neural.git
cd sparc-neural
pip install -e .
```

### Development installation
```bash
git clone https://github.com/yourusername/sparc-neural.git
cd sparc-neural
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from SPARC import InterpolationMethod, SPARCEvaluator

# Create some synthetic neural data
sampling_rate = 512
data = np.random.randn(1000, 5)  # 1000 samples, 5 channels

# Add some artifacts
data[100:150, :] += 10  # Simulate artifact

# Initialize and apply artifact removal
method = InterpolationMethod(sampling_rate, threshold=5.0)
cleaned_data = method.fit_transform(data)

# Evaluate the results
evaluator = SPARCEvaluator(sampling_rate)
metrics = evaluator.evaluate_method_comprehensive(ground_truth, original, cleaned_data)
print(f"MSE: {metrics['mse']:.6f}")
```

## Usage Examples

### Basic Artifact Removal
```python
from SPARC import TemplateSubtractionMethod, NeuralDataHandler

# Load your neural data
handler = NeuralDataHandler(sampling_rate=512)
data = handler.load_matlab_data('your_data.mat')

# Apply template subtraction
method = TemplateSubtractionMethod(sampling_rate=512, template_length_ms=200)
cleaned_data = method.fit_transform(data)
```

### Neural Signal Analysis
```python
from SPARC import NeuralAnalyzer

# Initialize analyzer
analyzer = NeuralAnalyzer(sampling_rate=512)

# Compute power spectral density
freqs, psd = analyzer.compute_psd(data)

# Extract neural features
features = analyzer.extract_neural_features(data, ['rms', 'variance', 'skewness'])

# Create comprehensive analysis plots
analyzer.plot_spectral_analysis(data, channel=0)
```

### Method Comparison
```python
from SPARC import InterpolationMethod, TemplateSubtractionMethod, SVDDecompositionMethod

# Test multiple methods
methods = {
    'Interpolation': InterpolationMethod(sampling_rate),
    'Template Subtraction': TemplateSubtractionMethod(sampling_rate),
    'SVD': SVDDecompositionMethod(sampling_rate)
}

# Compare performance
evaluator = SPARCEvaluator(sampling_rate)
results = evaluator.compare_methods(ground_truth, original, cleaned_signals)
```

## API Reference

### Core Classes
- `BaseSACMethod`: Abstract base class for all artifact removal methods
- `NeuralDataHandler`: Data loading and preprocessing
- `SPARCEvaluator`: Comprehensive evaluation metrics
- `NeuralAnalyzer`: Neural signal analysis tools

### Artifact Removal Methods
- `InterpolationMethod`: Interpolation-based artifact removal
- `TemplateSubtractionMethod`: Template subtraction
- `SVDDecompositionMethod`: SVD-based removal
- `DictionaryLearningMethod`: Dictionary learning (coming soon)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SPARC in your research, please cite:

```bibtex
@software{sparc2024,
  title={SPARC: Stimulation artifact Purging and Analysis in Real-time Circuits},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/sparc-neural}
}
```

## Support

- Documentation: [https://sparc-neural.readthedocs.io](https://sparc-neural.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/yourusername/sparc-neural/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/sparc-neural/discussions)