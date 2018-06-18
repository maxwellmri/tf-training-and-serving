import argparse
import os

import model

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

def run_experiment(hparams):
    # Create training and eval input functions
    train_input = lambda: model.generate_input_fn(
        args.train_files,
        num_epochs=hparams.num_epochs,
        batch_size=hparams.train_batch_size,
    )
    # Don't shuffle evaluation data
    eval_input = lambda: model.generate_input_fn(
        args.eval_files,
        batch_size=hparams.eval_batch_size,
        shuffle=False
    )

    # Define TrainSpec and EvalSpec instances

    # Define exporters for Eval Spec
    exporters = []
    exporters.append(tf.estimator.FinalExporter(
        'mnist', 
        model.example_serving_input_fn))

    eval_spec = tf.estimator.EvalSpec(
            eval_input, 
            steps=hparams.eval_steps, 
            exporters=exporters,
            throttle_secs=60)

    train_spec = tf.estimator.TrainSpec(
            train_input,
            max_steps=hparams.max_steps)

    # Create estimator
    estimator = tf.estimator.Estimator(
            model.generate_model_fn(hparams),
            config=tf.estimator.RunConfig(
                model_dir=hparams.job_dir,
            )
    )
    
    tf.estimator.train_and_evaluate(
        estimator, 
        train_spec, 
        eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-files',
        help='GCS or local paths to training data',
        nargs='+',
        required=True
    )
    parser.add_argument(
        '--num-epochs',
        help="""\
        Maximum number of training data epochs on which to train.
        If both --max-steps and --num-epochs are specified,
        the training job will run for --max-steps or --num-epochs,
        whichever occurs first. If unspecified will run for --max-steps.\
        """,
        type=int,
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=40
    )
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=40
    )
    parser.add_argument(
        '--eval-files',
        help='GCS or local paths to evaluation data',
        nargs='+',
        required=True
    )
    # Training arguments
    parser.add_argument(
        '--learning-rate',
        help='Learning rate for the optimizer',
        default=0.1,
        type=float
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='DEBUG',
        help='Set logging verbosity'
    )
    # Experiment arguments
    parser.add_argument(
        '--eval-delay-secs',
        help='How long to wait before running first evaluation',
        default=180,
        type=int
    )
    parser.add_argument(
        '--keep-prob',
        help='keep probability for dropout',
        default=0.5,
        type=int
    )
    parser.add_argument(
        '--min-eval-frequency',
        help='Minimum number of training steps between evaluations',
        default=100,
        type=int
    )
    parser.add_argument(
        '--max-steps',
        help="""\
        Steps to run the training job for. If --num-epochs is not specified,
        this must be. Otherwise the training job will run indefinitely.\
        """,
        type=int
    )
    parser.add_argument(
        '--eval-steps',
        help="""\
        Number of steps to run evalution for at each checkpoint.
        If unspecified will run until the input from --eval-files is exhausted
        """,
        default=None,
        type=int
    )
    parser.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='EXAMPLE'
    )

    args = parser.parse_args()
    hparams=hparam.HParams(**args.__dict__)

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    run_experiment(hparams)
