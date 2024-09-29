import tensorflow as tf
#from tensorflow.python.keras.layers import Dense, Flatten, Dropout
#import tensorflow.python.keras as tf
#pip install tf-keras
import keras

def list_keras_modules(module):
    print("Listing all modules and attributes in:", module.__name__)
    
    # List all attributes in the specified module
    attributes = dir(module)
    
    for attribute in attributes:
        # Get the attribute
        attr = getattr(module, attribute)
        # Check if it is a module or class
        if callable(attr) or isinstance(attr, type):
            print(f"- {attribute}")

# Explore the tensorflow.python.keras namespace
list_keras_modules(tf)