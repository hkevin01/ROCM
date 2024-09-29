import torch

def test_pytorch():
    # Check if PyTorch can be imported
    try:
        print("PyTorch is imported successfully.")
    except ImportError:
        print("Error: PyTorch is not installed.")
        return

    # Check ROCm availability
    if torch.cuda.is_available():
        device = 'cuda'
        print("ROCm is available. Using device:", torch.cuda.get_device_name(0))
    else:
        device = 'cpu'
        print("Neither ROCm nor CUDA is available. Using CPU.")

    # Perform a simple tensor operation
    try:
        # Create a tensor
        tensor = torch.tensor([[1, 2], [3, 4]], device=device)
        print("Created tensor:")
        print(tensor)

        # Perform an operation
        tensor_squared = tensor ** 2
        print("Tensor squared:")
        print(tensor_squared)

        # Check if the operation was successful
        assert tensor_squared.device.type == device, "Tensor operation failed to stay on the correct device."
        print("Tensor operation was successful.")
    except Exception as e:
        print(f"Error during tensor operations: {e}")

if __name__ == "__main__":
    test_pytorch()