#!/usr/bin/env python3
import subprocess
import sys
import os

def test_run_exp():
    print("Testing run_exp.sh script...")
    
    # Check if the script exists
    if not os.path.exists("run_exp.sh"):
        print("❌ run_exp.sh not found!")
        return False
    
    # Read the script to see what parameters it would use
    with open("run_exp.sh", "r") as f:
        script_content = f.read()
    
    print("✅ run_exp.sh found")
    print("Script content preview:")
    print("-" * 50)
    
    # Extract key parameters
    lines = script_content.split('\n')
    for line in lines:
        if any(keyword in line for keyword in ['regularization_strength_list', 'lambd_list', 'dataset_list', 'num_epochs', 'batch_size']):
            print(line.strip())
    
    print("-" * 50)
    
    # Test if we can run the Python command directly
    print("\nTesting direct Python command...")
    
    # Extract the command from the script
    cmd_parts = [
        "python", "main_MIA.py",
        "--arch=vgg11_bn_sgm",
        "--cutlayer=4", 
        "--batch_size=128",
        "--filename=test_run",
        "--num_client=1",
        "--num_epochs=2",  # Just 2 epochs for testing
        "--dataset=cifar10",
        "--scheme=V2_epoch",
        "--regularization=Gaussian_kl",
        "--regularization_strength=0.025",
        "--log_entropy=1",
        "--AT_regularization=SCA_new",
        "--AT_regularization_strength=0.3",
        "--random_seed=125",
        "--learning_rate=0.05",
        "--lambd=16",
        "--gan_AE_type=res_normN4C64",
        "--gan_loss_type=SSIM",
        "--local_lr=-1",
        "--bottleneck_option=noRELU_C8S1",
        "--folder=saves/cifar10/test_run",
        "--ssim_threshold=0.5",
        "--var_threshold=0.125",
        "--use_attention_classifier",
        "--num_slots=8",
        "--attention_heads=8",
        "--attention_dropout=0.1"
    ]
    
    print("Command to run:")
    print(" ".join(cmd_parts))
    
    try:
        # Run the command for just 2 epochs to test
        result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        print(f"\nExit code: {result.returncode}")
        print(f"STDOUT (first 1000 chars):")
        print(result.stdout[:1000])
        print(f"\nSTDERR (first 500 chars):")
        print(result.stderr[:500])
        
        if result.returncode == 0:
            print("✅ Test successful!")
            return True
        else:
            print("❌ Test failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

if __name__ == "__main__":
    test_run_exp()
