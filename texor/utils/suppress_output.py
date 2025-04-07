import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout_stderr():
    """
    Context manager để tạm thời chuyển hướng stdout và stderr.
    """
    # Lưu các file descriptors gốc
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    devnull = open(os.devnull, 'w')
    
    try:
        # Chuyển hướng stdout và stderr tới /dev/null
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        # Khôi phục stdout và stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()