from pynvml import *
import torch
from torch import cuda 
import math

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   i = 2
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])
   
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    used = info.used//1024**2
    total = info.total//1024**2
    print(f"GPU memory occupied: {used}MB ({((used / total) * 100):.2f}%)")
    print(f"Total GPU memory: {total}MB")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def print_gpu_info(device:torch.device = None):
    # cuda0 = torch.device("cuda:1")
    info_str = f"Printing device properties: \n{cuda.get_device_properties(device=device)}\n\n"
    info_str += f"Printing gpu processes: \n{cuda.list_gpu_processes(device=device)}\n\n"
    free, total =  cuda.mem_get_info(device)
    free, total = convert_size(free), convert_size(total)
    info_str += f"Printing memory info: \nfree:{free} \ntotal:{total}\n\n"
    #info_str += f"Printing memory stats: \n{cuda.memory_stats(device=device)}\n\n"
    #info_str += f"Printing memory summary: \n{cuda.memory_summary(device=device)}\n\n"
    #info_str += f"Printing memory snapshot: \n{cuda.memory_snapshot()}\n\n"
    print(info_str)



