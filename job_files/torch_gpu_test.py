import torch

print("Checking PyTorch setup...\n")

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check for CUDA
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# List GPUs
num_devices = torch.cuda.device_count()
print(f"Number of CUDA devices: {num_devices}")

if cuda_available and num_devices > 0:
    for i in range(num_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"  Memory cached: {torch.cuda.memory_reserved(i)} bytes")

    # Try a small tensor operation
    try:
        x = torch.tensor([1.0, 2.0]).cuda()
        y = x * 2
        print("\nGPU tensor operation successful. x*2 =", y)
    except Exception as e:
        print("\nGPU operation failed:", e)
else:
    print("\nNo usable CUDA devices found.")
