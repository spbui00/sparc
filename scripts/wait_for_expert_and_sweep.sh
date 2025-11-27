#!/bin/bash
# Save as: wait_for_expert_and_sweep.sh

EXPERT_DIR="/Users/bui/code/sparc/sparc-nn/saved_models"
SWEEP_SCRIPT="/Users/bui/code/sparc/sparc-nn/sweeps/hyperparameter_sweep_expert.py"

# Get the most recent expert model file before training
LAST_EXPERT=$(ls -t "$EXPERT_DIR"/neural_expert_*.pth 2>/dev/null | head -1)
LAST_MOD_TIME=$(stat -f "%m" "$LAST_EXPERT" 2>/dev/null || echo "0")

echo "Monitoring for new expert model..."
echo "Last expert model: $LAST_EXPERT"
echo "Waiting for new model to appear..."

# Wait for a new expert model file
while true; do
    NEWEST_EXPERT=$(ls -t "$EXPERT_DIR"/neural_expert_*.pth 2>/dev/null | head -1)
    
    if [ -n "$NEWEST_EXPERT" ]; then
        NEW_MOD_TIME=$(stat -f "%m" "$NEWEST_EXPERT" 2>/dev/null || echo "0")
        
        # Check if we have a newer file
        if [ "$NEW_MOD_TIME" -gt "$LAST_MOD_TIME" ]; then
            echo "New expert model detected: $NEWEST_EXPERT"
            echo "Waiting 10 seconds to ensure file is fully written..."
            sleep 10
            break
        fi
    fi
    
    sleep 30  # Check every 30 seconds
    echo "$(date): Still waiting for expert training to complete..."
done

echo "Expert training complete! Starting hyperparameter sweep..."
cd "$(dirname "$SWEEP_SCRIPT")"
python hyperparameter_sweep_expert.py
