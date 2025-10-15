# Artifact Removal Performance Analysis

## Results Summary

### Simulated Data Results

| Method | SNR Improvement (dB) | LFP PSD Correlation |
|--------|---------------------|-------------------|
| PCA    | 11.7184            | 0.7042            |
| ICA    | 4.1967             | 0.8525            |

### SWEC-ETHZ Data Results

| Method | SNR Improvement (dB) | LFP PSD Correlation |
|--------|---------------------|-------------------|
| PCA    | 15.8075            | 0.4717            |
| ICA    | 16.5748            | 0.8400            |

## Analysis

### Simulated Data
- **PCA** shows higher SNR improvement (11.72 dB) but lower LFP preservation (0.70 correlation)
- **ICA** shows lower SNR improvement (4.20 dB) but better LFP preservation (0.85 correlation)

### SWEC-ETHZ Data
- **ICA** shows both higher SNR improvement (16.57 dB) and better LFP preservation (0.84 correlation)
- **PCA** shows good SNR improvement (15.81 dB) but poor LFP preservation (0.47 correlation)

## Methodology

### LFP Extraction

The Local Field Potential (LFP) is extracted using the `extract_lfp` method in `NeuralAnalyzer`:

```python
def extract_lfp(self, data: np.ndarray, cutoff_freq: float = 200.0) -> np.ndarray:
    fs = float(np.asarray(self.sampling_rate).item())
    sos = signal.butter(4, cutoff_freq, btype='low', fs=fs, output='sos')
    lfp_data = signal.sosfiltfilt(sos, data, axis=2)
    return lfp_data
```

**Process:**
1. **Input validation**: Ensures data is 3D (trials, channels, timesteps)
2. **Low-pass filtering**: Applies a 4th-order Butterworth low-pass filter with 200 Hz cutoff frequency
3. **Bidirectional filtering**: Uses `sosfiltfilt` for zero-phase filtering to avoid phase distortion
4. **Output**: Returns filtered LFP data maintaining the original 3D structure

### LFP Evaluation

The LFP quality is evaluated using Power Spectral Density (PSD) correlation:

```python
def evaluate_lfp(self, cleaned_signal: np.ndarray, ground_truth_signal: np.ndarray) -> Dict[str, float]:
    lfp_cleaned = self.extract_lfp(cleaned_signal)
    lfp_ground_truth = self.extract_lfp(ground_truth_signal)

    freqs_gt, psd_gt = self.compute_psd(lfp_ground_truth)
    freqs_cleaned, psd_cleaned = self.compute_psd(lfp_cleaned)

    if np.std(psd_gt) > 1e-9 and np.std(psd_cleaned) > 1e-9:
        correlation = np.corrcoef(psd_gt.flatten(), psd_cleaned.flatten())[0, 1]
    else:
        correlation = np.nan
        
    return {'lfp_psd_correlation': correlation}
```

### SNR Calculation

Signal-to-Noise Ratio is calculated using the `calculate_snr` method:

```python
def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power == 0:
        return np.inf
        
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db
```

### SNR Improvement Calculation

SNR improvement is calculated as the difference between SNR after and before cleaning:

```python
def calculate_snr_improvement(self, original: np.ndarray, cleaned: np.ndarray, 
                             ground_truth: np.ndarray) -> float:
    # Calculate noise before cleaning
    noise_before = original - ground_truth
    snr_before = self.calculate_snr(ground_truth, noise_before)
    
    # Calculate noise after cleaning
    noise_after = cleaned - ground_truth
    snr_after = self.calculate_snr(ground_truth, noise_after)
    
    return snr_after - snr_before
```

