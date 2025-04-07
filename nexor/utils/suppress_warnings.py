import os
import warnings
import logging
import tensorflow as tf

def suppress_tensorflow_warnings():
    """Tắt các cảnh báo không cần thiết từ TensorFlow"""
    # Tắt cảnh báo từ TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel(logging.ERROR)
    
    # Tắt các cảnh báo Python
    warnings.filterwarnings('ignore')
    
    # Tắt thông báo oneDNN
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def enable_warnings():
    """Bật lại các cảnh báo nếu cần"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    tf.get_logger().setLevel(logging.INFO)
    warnings.resetwarnings()
    os.environ.pop('TF_ENABLE_ONEDNN_OPTS', None)