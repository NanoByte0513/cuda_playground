import torch
import numpy as np
from io import BytesIO

SEQ_LEN = 64
HIDDEN_SIZE = 1024

path1 = r"/home/wuyou/cuda_playground/try_cutlass/tensor1_fp16_16_8.bin"
path2 = r"/home/wuyou/cuda_playground/try_cutlass/tensor2_fp16_16_8.bin"

def save_tensors(m:int, n:int, k:int, j:int, path1=path1, path2=path2):
    tensor1 = torch.randn((m, n), dtype=torch.float16)
    tensor2 = torch.randn((k, j), dtype=torch.float16)

    # print(tensor1)
    # print(tensor2)
    print(torch.matmul(tensor1, tensor2.T))

    buffer = BytesIO()
    buffer.write(tensor1.numpy().tobytes())
    with open(path1, 'wb') as f:
        f.write(buffer.getvalue())

    buffer = BytesIO()
    buffer.write(tensor2.numpy().tobytes())
    with open(path2, 'wb') as f:
        f.write(buffer.getvalue())


def read_tensors(m:int, n:int, k:int, j:int, path1=path1, path2=path2):
    with open(path1, "rb") as f:
        tensor1_byte_data = f.read()
        np_arr1 = np.frombuffer(tensor1_byte_data, dtype=np.float16)
        tensor1 = torch.from_numpy(np_arr1).reshape((m, n))
    # print(tensor1)
    
    with open(path2, "rb") as f:
        tensor2_byte_data = f.read()
        np_arr2 = np.frombuffer(tensor2_byte_data, dtype=np.float16)
        tensor2 = torch.from_numpy(np_arr2).reshape((k, j))
    print(torch.matmul(tensor1, tensor2.T))

if __name__ == "__main__":
    save_tensors(16, 8, 8, 8)
    print("*"*50)
    # read_tensors(16, 8, 16, 8)