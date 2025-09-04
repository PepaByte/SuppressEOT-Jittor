import os
import atexit
import threading
import queue
import time
import csv
from typing import Dict, Optional, Any

def singleton(cls):
    """A decorator to implement the singleton pattern."""
    _instance = {}
    def _wrapper(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return _wrapper

# -------------- The singleton decorator is the same as above --------------

@singleton
class CSVLogger:
    """
    An asynchronous, singleton logger that writes metrics to CSV files.

    This logger uses a dedicated worker thread to write data, preventing I/O
    operations from blocking the main application thread. Each log category
    (e.g., 'gpu', 'iot') is written to its own separate CSV file.
    """
    def __init__(self,
                 log_dir: str = "./csv_logs",
                 run_name: str = None,
                 flush_every: int = 30):
        """
        Initializes the logger.

        Args:
            log_dir (str): The directory where CSV files will be saved.
            flush_every (int): How often (in seconds) to flush data to disk.
        """
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.flush_every = flush_every
        
        if run_name is None:
            self.run_name = time.strftime("%Y%m%d-%H%M%S")
        else:
            self.run_name = run_name

        # Dictionaries to hold file handles and csv writers for each log tag
        self._file_handlers = {}
        self._csv_writers = {}

        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        # Ensure cleanup happens when the program exits
        atexit.register(self._close)

    # ------------- Public API -------------
    def log_null_optim(self,
                       step: int,
                       iter: int,
                       loss: float,
                       timestamp: Optional[float] = None):
        """Logs null optimization metrics."""
        data = {
            "step": step,
            "iter": iter,
            "loss": loss,
            "time": timestamp
        }
        self._queue.put(("null_optim", data))

    def log_gpu(self, step: int, gpu_mem_mb: float):
        """Logs GPU memory usage."""
        data = {"step": step, "gpu_MB": gpu_mem_mb}
        self._queue.put(("memory", data))
        
    def log_ito(self, step: int, total_loss: float, timestamp: Optional[float] = None):
        """Logs ITO total loss and time per step."""
        data = {"step": step, "total_loss": total_loss, "time": timestamp}
        self._queue.put(("ito", data))          # tag 也同步改成 "ito"
    # --------------  Public API  --------------

    # ------------- Internal Implementation -------------
    def _worker_loop(self):
        """The main loop for the worker thread that processes log entries."""
        last_flush = time.time()
        while True:
            try:
                # Wait for a new item in the queue, with a timeout
                tag, data_dict = self._queue.get(timeout=1)

                # If this tag is new, create a new CSV file and writer
                if tag not in self._csv_writers:
                    filepath = os.path.join(self.log_dir, f"{self.run_name}_{tag}.csv")
                    # Use 'w' mode to create/truncate the file, newline='' is crucial for csv
                    file_handle = open(filepath, 'w', newline='', encoding='utf-8')
                    self._file_handlers[tag] = file_handle

                    # Create a DictWriter which uses dictionary keys as headers
                    writer = csv.DictWriter(file_handle, fieldnames=list(data_dict.keys()))
                    writer.writeheader()
                    self._csv_writers[tag] = writer

                # Write the data row
                self._csv_writers[tag].writerow(data_dict)
                self._queue.task_done() # Signal that the item has been processed

            except queue.Empty:
                # If the queue is empty, we do nothing and let the loop continue.
                # The flush check below will handle periodic flushing.
                pass

            # Periodically flush data to disk to ensure it's saved
            if time.time() - last_flush > self.flush_every:
                self._flush_all()
                last_flush = time.time()

    def _flush_all(self):
        """Flushes all open file handlers."""
        for f in self._file_handlers.values():
            f.flush()

    def _close(self):
        """Waits for the queue to be empty, then flushes and closes all files."""
        # Wait until all items in the queue have been processed
        self._queue.join()
        
        print("Closing logger: Flushing and closing all CSV files...")
        self._flush_all()
        for f in self._file_handlers.values():
            f.close()
        print("Logger closed.")

logger = CSVLogger()
