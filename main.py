# Copyright 2018 Wisdom D'Almeida
# Licensed under the Apache License, Version 2.0 (the "License")

from __future__ import absolute_import, division, print_function

#!pip install tf-nightly-2.0-preview
import tensorflow as tf
tf.enable_eager_execution()

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import functools

import prepare_dataset
import model

print("TensorFlow version:", tf.__version__)

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not"
    "specified, it can be automatically detected from metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, it can be automatically detected from metadata.")

# Model specific parameters
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
tf.flags.DEFINE_integer("iterations_per_loop", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS

MAX_PARAGRAPH_LENGTH = 5 # Fixed max number of sentences per report. This value comes from preprocessing
MAX_SENTENCE_LENGTH = 18 # Fixed max number of words per sentence. This value comes from preprocessing

"""
Download the Indiana University [Chest X-Ray dataset](https://openi.nlm.nih.gov/faq.php) to train our model.
The dataset contains 3955 chest radiology reports from various hospital systems and 7470 associated chest x-rays
(most reports are associated with 2 or more images representing frontal and lateral views).
"""
prepare_dataset.maybe_download()

"""
Preprocess the dataset
The dataset comes with PNG images (the chest x-rays) and XML files (the radiology reports).
This code parses the XML files and builds vectors associating each report with exactly one image.
A report is associated with 1 or more chest x-rays but we decide to consider only one.

Key information we extract from radiology reports are:

*   **Findings:** relates to observations regarding each part of the Chest examined.
*   **Impression:** a diagnostic based on the findings reported.
"""
all_findings, all_impressions, all_img_names, rids = prepare_dataset.extract_data()


all_findings, all_impressions, all_img_names, rids = shuffle(all_findings,
                                                             all_impressions,
                                                             all_img_names,
                                                             rids,
                                                             random_state=1)

# Initialize InceptionV3 and load the pretrained Imagenet weights
inception_model = prepare_dataset.init_inception_model()

# Preprocess and tokenize Findings and Impressions
tokenizer, findings_vector = prepare_dataset.transform_input(all_findings, all_impressions, MAX_PARAGRAPH_LENGTH, MAX_SENTENCE_LENGTH)

# Create training and validation sets using 80-20 split
img_name_train, img_name_test, findings_train, findings_test = train_test_split(all_img_names, findings_vector, test_size = 0.2, random_state = 0)

trainer = model.Trainer(tokenizer, embedding_dim=256, units=512)

FEATURES_SHAPE = 2048
ATTENTION_FEATURES_SHAPE = 64

#encode_train = sorted(set(all_img_names))
# feel free to change the batch_size according to your system configuration
#image_dataset = tf.data.Dataset.from_tensor_slices(
#                                encode_train).map(load_image).batch(64)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize_images(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def map_func(img_name, findings):
    img, img_path = load_image(img_name)
    img = tf.expand_dims(img, 0)
    img_tensor = inception_model(img)
    img_tensor = tf.reshape(img_tensor,
                            (-1, img_tensor.shape[3]))
    return img_tensor, findings

def _set_shapes(images, findings):
    # Statically set tensors dimensions
    #print(images.get_shape())
    images.set_shape(tf.TensorShape([ATTENTION_FEATURES_SHAPE, FEATURES_SHAPE]))
    findings.set_shape(findings.get_shape().merge_with(
            tf.TensorShape([MAX_PARAGRAPH_LENGTH + MAX_PARAGRAPH_LENGTH, MAX_SENTENCE_LENGTH])))
    return images, findings

def input_fn(params):
    batch_size = params['batch_size']
    #_img_name_train = np.asarray(img_name_train)
    _findings_train = np.asarray(findings_train)

    #my_dict = {
        #"img_tensors": _img_name_train,
        #"findings": _findings_train,
    #}

    #dataset = tf.data.Dataset.from_tensor_slices((dict(my_dict)))
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, _findings_train))

    # using map to load the numpy files in parallel
    # NOTE: Be sure to set num_parallel_calls to the number of CPU cores you have
    # https://www.tensorflow.org/api_docs/python/tf/py_func

    #dataset = dataset.map(lambda item: map_func, num_parallel_calls=8)
    #dataset = dataset.map(lambda item1, item2: tf.py_func(
            #map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=FLAGS.num_shards)

    dataset = dataset.map(map_func)

    #dataset = dataset.map(functools.partial(_set_shapes))

    # shuffling and batching
    dataset = dataset.shuffle(10000).repeat()
    # https://www.tensorflow.org/api_docs/python/tf/contrib/data/batch_and_drop_remainder
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)
    print("Dataset type:", dataset.output_shapes, dataset.output_types)
    return dataset

def model_fn(features, labels, mode, params):

    print("Model_Fn Shapes:", features.shape, labels.shape)
    print("Features:", features)
    batch_size = params['batch_size']

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
        loss, gradients, variables = trainer.train_fn(batch_size, features, labels)
        train_op = optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

        return tf.contrib.tpu.TPUEstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                               loss = loss,
                                               train_op = train_op)

def main(argv):
    del argv  # Unused.
    tf.logging.set_verbosity(tf.logging.INFO)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project
    )

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations_per_loop, FLAGS.num_shards),
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.batch_size,
        config=run_config
    )

    # TPUEstimator.train *requires* a max_steps argument.
    estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)

if __name__ == "__main__":
    tf.app.run()