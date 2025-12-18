import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version (torch): {torch.version.cuda}")
print(f"CUDNN Version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    try:
        print("Attempting tensor operation on CUDA...")
        x = torch.tensor([1.0]).cuda()
        print(f"Success: {x}")
        
        print("Attempting matrix multiplication...")
        a = torch.randn(100, 100).cuda()
        b = torch.randn(100, 100).cuda()
        c = torch.matmul(a, b)
        print("Success: Matrix mult complete")
    except Exception as e:
        print(f"FAILURE: {e}")
else:
    print("CUDA is NOT available.")
