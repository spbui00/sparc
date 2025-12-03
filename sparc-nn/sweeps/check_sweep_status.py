import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sweeps.generate_sweep_name import config_to_string

SWEEP_DIR = "sweep_res"

def config_exists(config_str):
    config_dir = os.path.join(SWEEP_DIR, config_str)
    results_file = os.path.join(config_dir, "results.txt")
    return os.path.exists(results_file)

hyperparameter_ranges = {
    'f_cutoff': [0, 10,25, 50],
    'w_cosine': [0, 1, 2],
    'w_rank_s': [0, 0, 0.2, 0.5],
    'w_spectral': [0, 1,2,3],
    'w_spectral_slope': [0,0.2,0.5,1],
    'w_rank_a': [0, 1,2,3],
}

configs = []
for f_cutoff in hyperparameter_ranges['f_cutoff']:
    for w_cosine in hyperparameter_ranges['w_cosine']:
            for w_rank_s in hyperparameter_ranges['w_rank_s']:
                for w_spectral in hyperparameter_ranges['w_spectral']:
                    for w_spectral_slope in hyperparameter_ranges['w_spectral_slope']:
                        for w_rank_a in hyperparameter_ranges['w_rank_a']:
                            configs.append({
                                'f_cutoff': f_cutoff,
                                'w_cosine': w_cosine,
                                'w_rank_s': w_rank_s,
                                'w_spectral': w_spectral,
                                'w_spectral_slope': w_spectral_slope,
                                'w_rank_a': w_rank_a,
                            })

configs.append({
    'f_cutoff': 50,
    'w_cosine': 5,
    'w_rank_s': 1,
    'w_spectral': 1.2,
    'w_spectral_slope': 5,
    'w_rank_a': 1.5,
    'use_uncertainty_loss': True
})

completed = 0
pending = []

for i, config in enumerate(configs):
    config_copy = config.copy()
    use_uncertainty = config_copy.pop('use_uncertainty_loss', False)
    config_str = config_to_string(config_copy, use_uncertainty_loss=use_uncertainty)
    
    if config_exists(config_str):
        completed += 1
    else:
        pending.append((i+1, config_str, config_copy, use_uncertainty))

total = len(configs)
remaining = total - completed

print(f"\n{'='*80}")
print(f"HYPERPARAMETER SWEEP STATUS")
print(f"{'='*80}")
print(f"Total configurations: {total}")
print(f"Completed: {completed} ({100*completed/total:.1f}%)")
print(f"Remaining: {remaining} ({100*remaining/total:.1f}%)")
print(f"{'='*80}")

if pending:
    print(f"\nFirst 10 pending configurations:")
    for idx, config_str, config, use_unc in pending[:10]:
        print(f"  {idx}. {config_str}")
    if len(pending) > 10:
        print(f"  ... and {len(pending) - 10} more")
else:
    print("\nAll configurations completed!")

print(f"\n{'='*80}\n")

