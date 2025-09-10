import numpy as np 
import scipy.io
import matplotlib.pyplot as plt


def load_data(file_path):
    try:
        data = scipy.io.loadmat(file_path) 

        gt = data['SimBB'].squeeze()
        artifacts = data['SimArtifact'].squeeze()
        mixed = data['SimCombined'].squeeze()

        sampling_rate = 30000 # read from matlab script

        return {
            'ground_truth': gt,
            'artifacts': artifacts,
            'mixed_data': mixed,
            'sampling_rate': sampling_rate
        }
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure the .mat file is in the same directory as this script.")
        return None
    except KeyError as e:
        print(f"Error: A required key is missing from the .mat file: {e}")
        return None


def plot_simulated_data(data, time_window_ms=50):
    if data is None:
        print("Cannot plot, data is not loaded.")
        return

    ground_truth = data['ground_truth']
    artifacts = data['artifacts']
    mixed_data = data['mixed_data']
    sampling_rate = data['sampling_rate']

    # Convert time window to number of samples
    num_samples = int(time_window_ms / 1000 * sampling_rate)
    time_axis = np.arange(num_samples) / sampling_rate * 1000  # Time in ms

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each component
    ax.plot(time_axis, ground_truth[:num_samples], label='Ground Truth (SimBB)', color='blue', linewidth=1.5)
    ax.plot(time_axis, artifacts[:num_samples], label='Artifact (SimArtifact)', color='red', alpha=0.6, linestyle='--')
    ax.plot(time_axis, mixed_data[:num_samples], label='Mixed Data (SimCombined)', color='black', alpha=0.7, linewidth=2)

    # Formatting
    ax.set_title(f'Simulated Data (First {time_window_ms} ms)', fontsize=16)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Amplitude (uV)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, time_window_ms)

    print("\nDisplaying plot...")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    simulated_data = load_data('../../research/generate_dataset/SimulatedData_1.mat')

    if simulated_data:
        plot_simulated_data(simulated_data, time_window_ms=50)
