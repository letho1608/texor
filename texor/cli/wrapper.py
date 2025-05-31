#!/usr/bin/env python
import os
import sys

def run_cli():
    # Set environment variables for optimal performance
    os.environ['NUMBA_THREADING_LAYER'] = 'threadsafe'
    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Redirect stdout/stderr to null for imports
    _stdout = sys.stdout
    _stderr = sys.stderr
    null = open(os.devnull, 'w')
    sys.stdout = null
    sys.stderr = null

    # Configure warnings for native backend
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # Restore stdout/stderr
    sys.stdout = _stdout
    sys.stderr = _stderr
    null.close()

    # Import and run the CLI
    from .main import main
    main()

if __name__ == '__main__':
    run_cli()