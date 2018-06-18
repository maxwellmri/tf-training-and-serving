import tensorflow as tf
from mnist_deep import conv2d, max_pool_2x2, weight_variable, bias_variable


def deepnn(x, keep_prob):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
    Returns:
    y. y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9).
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv

def calculate_loss(labels, logits):

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.argmax(labels,1),
            logits=logits)
    cross_entropy = tf.reduce_mean(cross_entropy)

    return cross_entropy

def get_eval_metrics(labels, logits):
    
    eval_metrics = {
                'accuracy': tf.metrics.accuracy(
                    tf.argmax(labels, 1),
                    tf.argmax(logits, 1))
    }

    return eval_metrics


def generate_model_fn(hparams):
    def _model_fn(mode, features, labels):
        image = features['image']
        
        y_conv = deepnn(image, hparams.keep_prob)
        prediction = tf.argmax(y_conv, 1)

        if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            loss = calculate_loss(labels, y_conv)

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                train_op = tf.train.AdamOptimizer(hparams.learning_rate).minimize(
                        loss,
                        global_step=tf.train.get_global_step())
            
            eval_metrics = get_eval_metrics(labels, y_conv)
            
            # Add tensorboard summaries
            tf.summary.scalar('loss', loss)
            tf.summary.image('input', tf.reshape(image, [-1, 28, 28, 1]))
            export_outputs = None
        else:
            loss = None
            train_op = None
            eval_metrics = None
            export_outputs = {
                'labels': tf.estimator.export.PredictOutput({
                    'label': prediction,
                    'id': features['id'],
                })}

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metrics,
            predictions={
                'label': tf.nn.softmax(tf.cast(prediction,tf.float32))},
            export_outputs=export_outputs
            )
    return _model_fn


def generate_input_fn(
        filenames,
        num_epochs=None,
        shuffle=True,
        batch_size=32):
    """Generates features and labels for training or evaluation.
    """
    def parse_function(example):
        features = {
            'image': tf.FixedLenFeature(
                shape=[784],
                dtype=tf.float32),
            'label': tf.FixedLenFeature(
                shape=[10],
                dtype=tf.float32),
            'id': tf.FixedLenFeature(
                shape=[1],
                dtype=tf.float32)
        }

        return tf.parse_single_example(example, features=features)
    
    dataset = tf.data.TFRecordDataset(filenames).map(parse_function)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    
    return features, features.pop('label')


def example_serving_input_fn():
    """Build the serving inputs."""
    features = {
        'image': tf.FixedLenFeature(
            shape=[784],
            dtype=tf.float32),
        'label': tf.FixedLenFeature(
            shape=[10],
            dtype=tf.float32),
        'id': tf.FixedLenFeature(
            shape=[1],
            dtype=tf.float32),
    }
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(features)()


SERVING_FUNCTIONS = {
    'EXAMPLE': example_serving_input_fn
    }


