import torch
import numpy as np
from io import BytesIO
import os

M = 32
N = 16
K = 64

home_path = r"/home/wuyou/cuda_playground/test_cublass"

path1 = os.path.join(home_path, f"tensor1_fp16_{M}_{K}.bin")
path2 = os.path.join(home_path, f"tensor2_fp16_{K}_{N}.bin")
product_path = os.path.join(home_path, f"product_fp16_{M}_{N}.bin")

def save_tensors(m:int, n:int, k:int, path1=path1, path2=path2, trans_B=False):
    tensor1 = torch.randn((m, k), dtype=torch.float16)
    tensor2 = torch.randn((k, n), dtype=torch.float16)

    buffer = BytesIO()
    buffer.write(tensor1.numpy().tobytes())
    with open(path1, 'wb') as f:
        f.write(buffer.getvalue())

    buffer = BytesIO()
    buffer.write(tensor2.numpy().tobytes())
    with open(path2, 'wb') as f:
        f.write(buffer.getvalue())

    product = torch.matmul(tensor1, tensor2.T) if trans_B else torch.matmul(tensor1, tensor2)
    print(product)
    buffer = BytesIO()
    buffer.write(product.numpy().tobytes())
    with open(product_path, 'wb') as f:
        f.write(buffer.getvalue())


def read_tensors(m:int, n:int, k:int, j:int, path1=path1, path2=path2):
    with open(path1, "rb") as f:
        tensor1_byte_data = f.read()
        np_arr1 = np.frombuffer(tensor1_byte_data, dtype=np.float16)
        tensor1 = torch.from_numpy(np_arr1).reshape((m, n))
    print(tensor1)
    
    with open(path2, "rb") as f:
        tensor2_byte_data = f.read()
        np_arr2 = np.frombuffer(tensor2_byte_data, dtype=np.float16)
        tensor2 = torch.from_numpy(np_arr2).reshape((k, j))
    print(tensor2)
    print(torch.matmul(tensor1, tensor2.T))

if __name__ == "__main__":
    save_tensors(M, N, K)
    print("*"*50)
    # read_tensors(16, 16, 8, 16)