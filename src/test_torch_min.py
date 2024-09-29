import torch

def check_rocm():
    if torch.cuda.is_available():
        print("ROCm is available.")
        print("Using device:", torch.cuda.get_device_name(0))
    else:
        print("ROCm is not available.")

if __name__ == "__main__":
    check_rocm()