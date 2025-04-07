#!/usr/bin/env python
import os
import sys

def run_cli():
    # Set environment variables to suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Redirect stdout/stderr to null for imports
    _stdout = sys.stdout
    _stderr = sys.stderr
    null = open(os.devnull, 'w')
    sys.stdout = null
    sys.stderr = null

    # Import tensorflow and configure
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)

    # Restore stdout/stderr
    sys.stdout = _stdout
    sys.stderr = _stderr
    null.close()

    # Import and run the CLI
    from .main import main
    main()

if __name__ == '__main__':
    run_cli()