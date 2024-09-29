try:
    import tensorflow as tf
    print("TensorFlow is installed.")
    
    # Check if Keras is available within TensorFlow
    if hasattr(tf, 'keras'):
        print("Keras is available within TensorFlow.")
        print(f"TensorFlow version: {tf.__version__}")
    else:
        print("Keras is not available within TensorFlow.")
except ImportError:
    print("TensorFlow is not installed. Please install it using pip.")