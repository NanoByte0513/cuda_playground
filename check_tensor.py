import torch
import numpy as np
from io import BytesIO

SEQ_LEN = 64
HIDDEN_SIZE = 1024

def save_tensors(m:int, n:int, k:int, j:int, path1=r"./tensor1.bin", path2=r"./tensor2.bin"):
    tensor1 = torch.randn((m, n), dtype=torch.float16)
    tensor2 = torch.randn((k, j), dtype=torch.float16)

    print(tensor1)
    print(tensor2)

    buffer = BytesIO()
    buffer.write(tensor1.numpy().tobytes())
    with open(path1, 'wb') as f:
        f.write(buffer.getvalue())

    buffer = BytesIO()
    buffer.write(tensor2.numpy().tobytes())
    with open(path2, 'wb') as f:
        f.write(buffer.getvalue())


def read_tensors(m:int, n:int, k:int, j:int, path1=r"./tensor1.bin", path2=r"./tensor2.bin"):
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

if __name__ == "__main__":
    # save_tensors(SEQ_LEN, HIDDEN_SIZE, SEQ_LEN, HIDDEN_SIZE)
    print("*"*50)
    read_tensors(SEQ_LEN, HIDDEN_SIZE, SEQ_LEN, HIDDEN_SIZE)