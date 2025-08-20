#!/usr/bin/env python3

"""
Simple training script for CEM-att
"""

import subprocess
import sys
import os

def run_attention_training():
    """Run CEM training with attention classifier"""
    
    print("🚀 Starting CEM-att training with Attention classifier")
    print("=" * 60)
    
    # Training command with attention enabled
    cmd = [
        "python", "main_MIA.py",
        "--filename", "cem_attention_test",
        "--arch", "vgg11_bn",
        "--cutlayer", "4",
        "--batch_size", "64",
        "--num_epochs", "5",
        "--learning_rate", "0.01",
        "--lambd", "1.0",
        "--dataset", "cifar10",
        "--use_attention_classifier",  # Enable attention classifier
        "--num_slots", "8",
        "--attention_heads", "8",
        "--attention_dropout", "0.1",
        "--log_entropy", "1"  # Use log entropy for better training
    ]
    
    print("Command to run:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Training completed successfully!")
        print("\nTraining output:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print("\nError output:")
        print(e.stderr)
        print("\nStdout:")
        print(e.stdout)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def run_baseline_training():
    """Run CEM training without attention (baseline)"""
    
    print("🔄 Starting baseline CEM training (without attention)")
    print("=" * 60)
    
    # Training command without attention
    cmd = [
        "python", "main_MIA.py",
        "--filename", "cem_baseline_test",
        "--arch", "vgg11_bn",
        "--cutlayer", "4",
        "--batch_size", "64",
        "--num_epochs", "5",
        "--learning_rate", "0.01",
        "--lambd", "1.0",
        "--dataset", "cifar10",
        "--log_entropy", "1"
        # Note: No --use_attention_classifier flag
    ]
    
    print("Command to run:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Baseline training completed successfully!")
        print("\nTraining output:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Baseline training failed with exit code {e.returncode}")
        print("\nError output:")
        print(e.stderr)
        print("\nStdout:")
        print(e.stdout)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 CEM-att Training Test Suite")
    print("=" * 60)
    
    # Change to the correct directory
    os.chdir('/Users/asher/Documents/attention/CEM-att')
    
    success_attention = run_attention_training()
    print()
    
    if success_attention:
        print("🎉 Attention-based CEM training is working!")
        print("\n🚀 CEM-att implementation successfully integrates attention mechanism!")
        print("\nTo use attention classifier, run:")
        print("python main_MIA.py --filename your_experiment --use_attention_classifier --lambd 1.0")
    else:
        print("💥 Attention-based CEM training has issues.")
        
        # Try baseline training to isolate the problem
        print("\nTrying baseline training to isolate the issue...")
        success_baseline = run_baseline_training()
        
        if success_baseline:
            print("\n💡 Baseline training works, issue is with attention integration.")
        else:
            print("\n💥 Both attention and baseline training failed - check CEM-main setup.")
    
    sys.exit(0 if success_attention else 1)

