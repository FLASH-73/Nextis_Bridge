#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import multiprocessing
import queue
import threading
from pathlib import Path

import numpy as np
import PIL.Image
import torch


def safe_stop_image_writer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            dataset = kwargs.get("dataset")
            image_writer = getattr(dataset, "image_writer", None) if dataset else None
            if image_writer is not None:
                print("Waiting for image writer to terminate...")
                image_writer.stop()
            raise e

    return wrapper


def image_array_to_pil_image(image_array: np.ndarray, range_check: bool = True) -> PIL.Image.Image:
    # Handle 1-channel depth images and 3-channel RGB images
    if image_array.ndim != 3:
        raise ValueError(f"The array has {image_array.ndim} dimensions, but 3 is expected for an image.")

    # Check for single-channel (depth) images
    is_depth = image_array.shape[-1] == 1 or image_array.shape[0] == 1

    if is_depth:
        # Handle depth images (1 channel)
        if image_array.shape[0] == 1:
            # (1, H, W) -> (H, W)
            image_array = image_array.squeeze(0)
        elif image_array.shape[-1] == 1:
            # (H, W, 1) -> (H, W)
            image_array = image_array.squeeze(-1)

        # Depth is typically uint16 (millimeters) - save as 16-bit grayscale PNG
        if image_array.dtype == np.uint16:
            return PIL.Image.fromarray(image_array, mode="I;16")
        elif image_array.dtype == np.uint8:
            return PIL.Image.fromarray(image_array, mode="L")
        else:
            # Convert float depth to uint16 (assume meters, scale to mm)
            if image_array.dtype in (np.float32, np.float64):
                # Assume depth in meters, convert to mm and clamp to uint16 range
                depth_mm = (image_array * 1000).clip(0, 65535).astype(np.uint16)
                return PIL.Image.fromarray(depth_mm, mode="I;16")
            # Fallback: convert to uint8
            image_array = image_array.astype(np.uint8)
            return PIL.Image.fromarray(image_array, mode="L")

    # Handle RGB images (3 channels)
    if image_array.shape[0] == 3:
        # Transpose from pytorch convention (C, H, W) to (H, W, C)
        image_array = image_array.transpose(1, 2, 0)

    elif image_array.shape[-1] != 3:
        raise NotImplementedError(
            f"The image has {image_array.shape[-1]} channels, but 3 is required for RGB images."
        )

    if image_array.dtype != np.uint8:
        if range_check:
            max_ = image_array.max().item()
            min_ = image_array.min().item()
            if max_ > 1.0 or min_ < 0.0:
                raise ValueError(
                    "The image data type is float, which requires values in the range [0.0, 1.0]. "
                    f"However, the provided range is [{min_}, {max_}]. Please adjust the range or "
                    "provide a uint8 image with values in the range [0, 255]."
                )

        image_array = (image_array * 255).astype(np.uint8)

    return PIL.Image.fromarray(image_array)


def write_image(image: np.ndarray | PIL.Image.Image, fpath: Path, compress_level: int = 1):
    """
    Saves a NumPy array or PIL Image to a file.

    This function handles both NumPy arrays and PIL Image objects, converting
    the former to a PIL Image before saving. It includes error handling for
    the save operation.

    Args:
        image (np.ndarray | PIL.Image.Image): The image data to save.
        fpath (Path): The destination file path for the image.
        compress_level (int, optional): The compression level for the saved
            image, as used by PIL.Image.save(). Defaults to 1.
            Refer to: https://github.com/huggingface/lerobot/pull/2135
            for more details on the default value rationale.

    Raises:
        TypeError: If the input 'image' is not a NumPy array or a
            PIL.Image.Image object.

    Side Effects:
        Prints an error message to the console if the image writing process
        fails for any reason.
    """
    try:
        if isinstance(image, np.ndarray):
            img = image_array_to_pil_image(image)
        elif isinstance(image, PIL.Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        img.save(fpath, compress_level=compress_level)
    except Exception as e:
        print(f"Error writing image {fpath}: {e}")


def worker_thread_loop(queue: queue.Queue):
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break
        image_array, fpath, compress_level = item
        write_image(image_array, fpath, compress_level)
        queue.task_done()


def worker_process(queue: queue.Queue, num_threads: int):
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker_thread_loop, args=(queue,))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


class AsyncImageWriter:
    """
    This class abstract away the initialisation of processes or/and threads to
    save images on disk asynchronously, which is critical to control a robot and record data
    at a high frame rate.

    When `num_processes=0`, it creates a threads pool of size `num_threads`.
    When `num_processes>0`, it creates processes pool of size `num_processes`, where each subprocess starts
    their own threads pool of size `num_threads`.

    The optimal number of processes and threads depends on your computer capabilities.
    We advise to use 4 threads per camera with 0 processes. If the fps is not stable, try to increase or lower
    the number of threads. If it is still not stable, try to use 1 subprocess, or more.
    """

    def __init__(self, num_processes: int = 0, num_threads: int = 1):
        self.num_processes = num_processes
        self.num_threads = num_threads
        self.queue = None
        self.threads = []
        self.processes = []
        self._stopped = False

        if num_threads <= 0 and num_processes <= 0:
            raise ValueError("Number of threads and processes must be greater than zero.")

        if self.num_processes == 0:
            # Use threading
            self.queue = queue.Queue()
            for _ in range(self.num_threads):
                t = threading.Thread(target=worker_thread_loop, args=(self.queue,))
                t.daemon = True
                t.start()
                self.threads.append(t)
        else:
            # Use multiprocessing
            self.queue = multiprocessing.JoinableQueue()
            for _ in range(self.num_processes):
                p = multiprocessing.Process(target=worker_process, args=(self.queue, self.num_threads))
                p.daemon = True
                p.start()
                self.processes.append(p)

    def save_image(
        self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path, compress_level: int = 1
    ):
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy array to minimize main process time
            image = image.cpu().numpy()
        self.queue.put((image, fpath, compress_level))

    def wait_until_done(self, timeout: float = 60.0):
        """Wait for all queued images to be written.

        Args:
            timeout: Maximum seconds to wait. Default 60s. If exceeded, logs warning and returns.
        """
        import time
        import logging

        start = time.time()
        check_interval = 0.5  # Check every 500ms

        # Poll until queue is empty or timeout
        while True:
            # Check if queue is empty (all items processed)
            if self.queue.empty():
                # Give a brief moment for final task_done() calls
                time.sleep(0.1)
                if self.queue.empty():
                    break

            # Check timeout
            elapsed = time.time() - start
            if elapsed > timeout:
                remaining = self.queue.qsize()
                logging.warning(
                    f"[ImageWriter] wait_until_done() timed out after {timeout:.1f}s "
                    f"with ~{remaining} items still in queue. Proceeding anyway."
                )
                return  # Don't block forever, just proceed

            time.sleep(check_interval)

        # Queue is empty, do a final join to ensure all task_done() calls complete
        # Use a short timeout thread to avoid infinite hang
        import threading
        done_event = threading.Event()

        def final_join():
            try:
                self.queue.join()
            except Exception:
                pass
            done_event.set()

        join_thread = threading.Thread(target=final_join, daemon=True)
        join_thread.start()

        # Wait up to 5 seconds for final join
        if not done_event.wait(timeout=5.0):
            logging.warning("[ImageWriter] Final queue.join() timed out after 5s")

    def stop(self):
        if self._stopped:
            return

        if self.num_processes == 0:
            for _ in self.threads:
                self.queue.put(None)
            for t in self.threads:
                t.join()
        else:
            num_nones = self.num_processes * self.num_threads
            for _ in range(num_nones):
                self.queue.put(None)
            for p in self.processes:
                p.join()
                if p.is_alive():
                    p.terminate()
            self.queue.close()
            self.queue.join_thread()

        self._stopped = True
