# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Run ALBERT on SQuAD 1.1 and SQuAD 2.0. using sentence piece tokenization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import six

import tensorflow as tf
import tensorflow_hub as hub

import optimization
import run_squad_sp
import tokenization
import modeling
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu

# pylint: disable=g-import-not-at-top
if six.PY2:
  import six.moves.cPickle as pickle
else:
  import pickle
  

flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "albert_hub_module_handle", None,
    "The tf_hub file corresponding to the pre-trained ALBERT model.")

def create_tokenizer_from_hub_module(albert_hub_module_handle):
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    albert_module = hub.Module(albert_hub_module_handle)
    tokenization_info = albert_module(signature="tokenization_info",
                                      as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
  return tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case,
      spm_model_file=FLAGS.spm_model_file)

def create_model(is_training, input_ids, input_mask, segment_ids,
                 albert_hub_module_handle):
    
  tags = set()
  if is_training:
    tags.add("train")
  albert_module = hub.Module(albert_hub_module_handle, tags=tags,
                             trainable=True)
  final_hidden = albert_module.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  return (start_logits, end_logits)

def model_fn_builder(learning_rate, num_train_steps, num_warmup_steps, use_tpu,
                     albert_hub_module_handle):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_model(
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids)
    
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)

      total_loss = (start_loss + end_loss) / 2.0

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions)
    return output_spec

  return model_fn

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)


  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = create_tokenizer_from_hub_module(FLAGS.albert_hub_module_handle)
  
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
  run_config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=contrib_tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  
  if FLAGS.do_train:
    train_examples = run_squad_sp.read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(learning_rate = FLAGS.learning_rate,
                              num_train_steps =num_train_steps,
                              num_warmup_steps=num_warmup_steps,
                              use_tpu=FLAGS.use_tpu,
                              albert_hub_module_handle =FLAGS.albert_hub_module_handle)
  
  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = contrib_tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)
######################################################################

#  if FLAGS.do_train:
#    # We write to a temporary file to avoid storing very large constant tensors
#    # in memory.
#
#
#    if not tf.gfile.Exists(FLAGS.train_feature_file):
#      train_writer = run_squad_sp.FeatureWriter(
#          filename=os.path.join(FLAGS.train_feature_file), is_training=True)
#      run_squad_sp.convert_examples_to_features(
#          examples=train_examples,
#          tokenizer=tokenizer,
#          max_seq_length=FLAGS.max_seq_length,
#          doc_stride=FLAGS.doc_stride,
#          max_query_length=FLAGS.max_query_length,
#          is_training=True,
#          output_fn=train_writer.process_feature)
#      train_writer.close()
#
#    tf.logging.info("***** Running training *****")
#    tf.logging.info("  Num orig examples = %d", len(train_examples))
#    # tf.logging.info("  Num split examples = %d", train_writer.num_features)
#    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
#    tf.logging.info("  Num steps = %d", num_train_steps)
#    del train_examples
#
#    train_input_fn = run_squad_sp.input_fn_builder(
#        input_file=FLAGS.train_feature_file,
#        seq_length=FLAGS.max_seq_length,
#        is_training=True,
#        drop_remainder=True)
#    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    eval_examples = run_squad_sp.read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False)

    if (FLAGS.predict_feature_file and FLAGS.predict_feature_left_file):
      tf.logging.info("Loading eval features from {}".format(
         FLAGS.predict_feature_left_file))
      with tf.gfile.Open(FLAGS.predict_feature_left_file, 'rb') as fin:
        eval_features = pickle.load(fin)
    else:
      eval_writer = run_squad_sp.FeatureWriter(
          filename=os.path.join(FLAGS.output_dir, "train.tf_record"), is_training=False)
      eval_features = []

      def append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

      run_squad_sp.convert_examples_to_features(
          examples=eval_examples,
          tokenizer=tokenizer,
          max_seq_length=FLAGS.max_seq_length,
          doc_stride=FLAGS.doc_stride,
          max_query_length=FLAGS.max_query_length,
          is_training=False,
          output_fn=append_feature)
      eval_writer.close()

      with tf.gfile.Open(os.path.join(FLAGS.output_dir, "train.tf_record"), 'wb') as fout:
        pickle.dump(eval_features, fout)

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []

    predict_input_fn = run_squad_sp.input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
          RawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits))

    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    run_squad_sp.write_predictions(eval_examples, eval_features, all_results,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file)
  

if __name__ == "__main__":
  flags.mark_flag_as_required("albert_hub_module_handle")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
