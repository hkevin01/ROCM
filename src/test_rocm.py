import os
import subprocess
import torch

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

def print_rocm_info():
    # Get the number of devices
    device_count = torch.cuda.device_count()
    print(f"Number of ROCm devices: {device_count}")

    for device_id in range(device_count):
        device = torch.cuda.get_device_properties(device_id)

        print(f"\nDevice {device_id}:")
        print(f"  Name: {device.name}")
        print(f"  Total Memory: {device.total_memory / (1024 ** 2):.2f} MB")
        print(f"  Compute Capability: {device.major}.{device.minor}")
        print(f"  Max Clock Frequency: {device.clockRate / 1000:.2f} MHz")
        print(f"  Multi Processor Count: {device.multiProcessorCount}")

        # Additional information can be printed as needed
        print(f"  Max Threads Per Block: {device.maxThreadsPerBlock}")
        print(f"  Max Block Dimensions: {device.maxBlockDimX}, {device.maxBlockDimY}, {device.maxBlockDimZ}")

if __name__ == "__main__":
    if check_rocm_installation():
        print_gpu_info()
        print_rocm_info()
    else:
        print("Please install ROCm to use this script.")