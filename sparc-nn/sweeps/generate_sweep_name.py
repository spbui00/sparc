#!/usr/bin/env python3
"""
Script to generate sweep directory/file names from hyperparameter weights.
"""

def config_to_string(config, use_uncertainty_loss=False):
    """
    Generate a config string from hyperparameter dictionary.
    
    Args:
        config: Dictionary with hyperparameter values
        use_uncertainty_loss: Whether to append 'uncertainty_loss' to the name
    
    Returns:
        String like: f_cutoff_10_w_cosine_1_w_rank_a_0.20_w_rank_s_0_w_spectral_3_w_spectral_slope_0
    """
    parts = []
    for key, val in sorted(config.items()):
        if key == 'f_cutoff':
            parts.append(f"{key}_{int(val)}")
        elif isinstance(val, float):
            if val.is_integer():
                parts.append(f"{key}_{int(val)}")
            else:
                parts.append(f"{key}_{val:.2f}")
        else:
            parts.append(f"{key}_{val}")
    if use_uncertainty_loss:
        parts.append("uncertainty_loss")
    return "_".join(parts)

def generate_sweep_name(f_cutoff=None, w_cosine=None, w_rank_s=None, 
                        w_spectral=None, w_spectral_slope=None, w_rank_a=None,
                        use_uncertainty_loss=False):
    """
    Generate sweep directory name from individual hyperparameter values.
    
    Args:
        f_cutoff: Frequency cutoff (Hz)
        w_cosine: Weight for cosine similarity loss
        w_rank_s: Weight for neural signal rank penalty
        w_spectral: Weight for spectral loss
        w_spectral_slope: Weight for spectral slope loss
        w_rank_a: Weight for artifact rank penalty
        use_uncertainty_loss: Whether to use uncertainty-weighted loss
    
    Returns:
        Directory name string
    """
    config = {}
    
    if f_cutoff is not None:
        config['f_cutoff'] = f_cutoff
    if w_cosine is not None:
        config['w_cosine'] = w_cosine
    if w_rank_s is not None:
        config['w_rank_s'] = w_rank_s
    if w_spectral is not None:
        config['w_spectral'] = w_spectral
    if w_spectral_slope is not None:
        config['w_spectral_slope'] = w_spectral_slope
    if w_rank_a is not None:
        config['w_rank_a'] = w_rank_a
    
    return config_to_string(config, use_uncertainty_loss=use_uncertainty_loss)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate sweep directory/file names from hyperparameter weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate name for specific configuration
  python generate_sweep_name.py --f-cutoff 10 --w-cosine 1 --w-rank-s 0 --w-spectral 3 --w-spectral-slope 0 --w-rank-a 0.2
  
  # Generate name with uncertainty loss
  python generate_sweep_name.py --f-cutoff 50 --w-cosine 5 --w-rank-s 1 --w-spectral 1.2 --w-spectral-slope 5 --w-rank-a 1.5 --use-uncertainty-loss
  
  # Generate name from config dict (interactive)
  python generate_sweep_name.py --interactive
        """
    )
    
    parser.add_argument('--f-cutoff', type=float, default=None,
                        help='Frequency cutoff (Hz)')
    parser.add_argument('--w-cosine', type=float, default=None,
                        help='Weight for cosine similarity loss')
    parser.add_argument('--w-rank-s', type=float, default=None,
                        help='Weight for neural signal rank penalty')
    parser.add_argument('--w-spectral', type=float, default=None,
                        help='Weight for spectral loss')
    parser.add_argument('--w-spectral-slope', type=float, default=None,
                        help='Weight for spectral slope loss')
    parser.add_argument('--w-rank-a', type=float, default=None,
                        help='Weight for artifact rank penalty')
    parser.add_argument('--use-uncertainty-loss', action='store_true',
                        help='Append uncertainty_loss to the name')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode: enter values one by one')
    
    args = parser.parse_args()
    
    if args.interactive:
        print("Interactive mode: Enter hyperparameter values (press Enter to skip)")
        print("="*60)
        
        f_cutoff = input("f_cutoff (Hz): ").strip()
        f_cutoff = float(f_cutoff) if f_cutoff else None
        
        w_cosine = input("w_cosine: ").strip()
        w_cosine = float(w_cosine) if w_cosine else None
        
        w_rank_s = input("w_rank_s: ").strip()
        w_rank_s = float(w_rank_s) if w_rank_s else None
        
        w_spectral = input("w_spectral: ").strip()
        w_spectral = float(w_spectral) if w_spectral else None
        
        w_spectral_slope = input("w_spectral_slope: ").strip()
        w_spectral_slope = float(w_spectral_slope) if w_spectral_slope else None
        
        w_rank_a = input("w_rank_a: ").strip()
        w_rank_a = float(w_rank_a) if w_rank_a else None
        
        use_uncertainty = input("Use uncertainty loss? (y/n): ").strip().lower() == 'y'
        
        sweep_name = generate_sweep_name(
            f_cutoff=f_cutoff,
            w_cosine=w_cosine,
            w_rank_s=w_rank_s,
            w_spectral=w_spectral,
            w_spectral_slope=w_spectral_slope,
            w_rank_a=w_rank_a,
            use_uncertainty_loss=use_uncertainty
        )
    else:
        if all(v is None for v in [args.f_cutoff, args.w_cosine, args.w_rank_s, 
                                    args.w_spectral, args.w_spectral_slope, args.w_rank_a]):
            print("Error: At least one hyperparameter must be provided.")
            print("Use --interactive for interactive mode or provide values via flags.")
            parser.print_help()
            return
        
        sweep_name = generate_sweep_name(
            f_cutoff=args.f_cutoff,
            w_cosine=args.w_cosine,
            w_rank_s=args.w_rank_s,
            w_spectral=args.w_spectral,
            w_spectral_slope=args.w_spectral_slope,
            w_rank_a=args.w_rank_a,
            use_uncertainty_loss=args.use_uncertainty_loss
        )
    
    print("\n" + "="*60)
    print("Generated Sweep Name:")
    print("="*60)
    print(sweep_name)
    print("="*60)
    print(f"\nFull path: sweep_1/{sweep_name}/")
    print(f"Results file: sweep_1/{sweep_name}/results.txt")
    print(f"Model file: sweep_1/{sweep_name}/model.pth")

if __name__ == '__main__':
    main()

