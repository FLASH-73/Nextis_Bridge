import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import time

class SharedMemoryRingBuffer:
    def __init__(self, name, n_slots, frame_shape, dtype=np.uint8, create=False):
        self.name = name
        self.n_slots = n_slots
        self.frame_shape = frame_shape # (H, W, C)
        self.dtype = dtype
        self.frame_size = int(np.prod(frame_shape)) * np.dtype(dtype).itemsize
        self.total_size = self.frame_size * n_slots
        
        if create:
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=self.total_size, name=name)
            except FileExistsError:
                self.shm = shared_memory.SharedMemory(create=False, name=name)
        else:
             self.shm = shared_memory.SharedMemory(create=False, name=name)
             
        # Create numpy array wrapper
        # We model it as (N, H, W, C)
        full_shape = (n_slots,) + frame_shape
        self.buffer = np.ndarray(full_shape, dtype=dtype, buffer=self.shm.buf)
        
    def write(self, slot_idx, data):
        """Writes data to slot. Data must match shape."""
        # This is a direct memcpy in C
        self.buffer[slot_idx] = data
        
    def read(self, slot_idx):
        """Returns reference (copy if needed) to slot."""
        return self.buffer[slot_idx] # Returns view
        
    def close(self):
        self.shm.close()
        
    def unlink(self):
        self.shm.unlink()
