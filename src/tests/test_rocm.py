import os
import subprocess

def check_rocm_installation():
    try:
        # Check if ROCm is installed by looking for ROCm utilities
        result = subprocess.run(["/opt/rocm/bin/rocm-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("ROCm is installed.")
            return True
        else:
            print("ROCm is not installed.")
            return False
    except FileNotFoundError:
        print("ROCm is not installed. 'rocm-smi' command not found.")
        return False

def print_gpu_info():
    try:
        # Get GPU information using rocm-smi
        result = subprocess.run(["/opt/rocm/bin/rocm-smi"], stdout=subprocess.PIPE, text=True)
        print("GPU Information:")
        print(result.stdout)
    except Exception as e:
        print(f"Error retrieving GPU information: {e}")


if __name__ == "__main__":
    if check_rocm_installation():
        print_gpu_info()
    else:
        print("Please install ROCm to use this script.")