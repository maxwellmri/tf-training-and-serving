from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import predict_pb2
import logging
from tensorflow.python.saved_model import signature_constants
import requests
import tensorflow as tf


class ServableClient():
    def __init__(self,
                host,
                model_name,
                port=9000,
                signature_name=signature_constants.
                DEFAULT_SERVING_SIGNATURE_DEF_KEY):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.signature_name = signature_name
        channel = implementations.insecure_channel(host, port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
           channel)

    def do_inference(self, inputs, timeout=600.0):
        example = self.create_example(inputs)
        try:
            request = predict_pb2.PredictRequest()
        except requests.exceptions.HTTPError as e:
            logging.warning("Error %s occured, while requesting a job." % e)
            raise

        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.signature_name
        logging.info('Starting inference...')
        
        request.inputs['examples'].CopyFrom(
           tf.contrib.util.make_tensor_proto([example], dtype=tf.string))
        result = self.stub.Predict(request, timeout)  # 60 second timeout
        logging.info('Finished inference.')
        return result

    def create_example(self, features):
        example = tf.train.Example(features=features)
        example_serialized = example.SerializeToString()
        return example_serialized


def create_features(image, label, _id):
    """
    Creates features for inference
    Args:
       image: numpy ndarray of shape [784,]
       label: one hot encoded labels of shape [10,]
       id: unique id for the image of shape [1,]
    Returns:
       tensorflow tf.train.Feature
    """

    features = {
       "image":
       tf.train.Feature(
           float_list=tf.train.FloatList(value=image.flatten().tolist())),
       "label":
       tf.train.Feature(
           float_list=tf.train.FloatList(value=label.flatten().tolist())),
       "id":
       tf.train.Feature(
           float_list=tf.train.FloatList(value=[_id])),
    }
    features = tf.train.Features(feature=features)

    return features
