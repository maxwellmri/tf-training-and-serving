import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from servable_client import ServableClient, create_features

def main():

    # Read mnist validation dataset
    mnist = input_data.read_data_sets("data/", one_hot=True)
    val = mnist.validation.next_batch(10)

    # Create example features
    features = []
    i = 0
    for img, label in zip(val[0], val[1]):
        features.append(create_features(img, label, i))
        i=+1
    
    # Initialise Client
    servable_client = ServableClient("10.10.0.10", "mnist-serving", port=9001)

    # Run Inference
    results = []
    for feature in features:
        results.append(servable_client.do_inference(feature).outputs['label'].int64_val[0])
    
    # Print Results
    for label, pred in zip(val[1], results):
        print('Predicted: %d'%pred, 'Label: %d'%np.argmax(label))

if __name__ == '__main__':
    main()
