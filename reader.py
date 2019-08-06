from __future__ import absolute_import, division, unicode_literals

import multiprocessing as mp

import tensorflow as tf

from common.train import inputs
from common.util import file_util, mx_logging
import numpy as np
import projects.edison.constants as ec
import random
import sys
from projects.edison.train.model_preprocessors.exam_image_splitting import ExamToImagesSplitterModelPreprocessor
from projects.edison.util.notebook_util import get_feature_dicts
# import utils.py
from projects.edison.train import util
from six import iteritems
_cached_converter = None
_cached_output_dim = None

class Reader():
    def __init__(self, tfrecords_file, image_width=1152, image_height=960,
                 min_queue_examples=1000, batch_size=1, num_threads=8, name=''):
        """
        Args:
          tfrecords_file: string, tfrecords file path
          min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
          batch_size: integer, number of images per batch
          num_threads: integer, number of preprocess threads
        """
        self.tfrecords_file = tfrecords_file
        self.image_width = image_width
        self.image_height = image_height
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.output_dim = (image_width, image_height)
        self.name = name

    def convert2float(self, image):
        """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
        """
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = (image / 127.5) - 1.0
        image = tf.tile(tf.expand_dims(image, -1), [1, 1, 3])

        # image.set_shape([self.image_width, self.image_height, 3])
        return image

    def _get_input_fn(self, path_globs, from_exam_converter):
        global _cached_converter, _cached_output_dim
        if _cached_converter is None:
            _cached_converter, _cached_output_dim = util.get_preprocessing_params(path_globs)

        if from_exam_converter:
            model_preprocessors = [ExamToImagesSplitterModelPreprocessor(_cached_output_dim)]
        else:
            model_preprocessors = []

        paths = inputs.get_tfrecord_paths(path_globs)
        return inputs.input_fn_factory(
            tfrecord_fpaths=paths,
            feature_schema=_cached_converter.get_feature_schema(),
            batch_size=1,
            num_epochs=1,
            model_preprocessors=model_preprocessors,
        )

    def get_feature_dicts(self, path_globs, from_exam_converter=True):
        """
        Args:
            path_globs: List of filename globs, e.g.
                ["/merantix_core/data/mx-healthcare-derived/preprocessed/diranuk_local_224x192/all/*TRAIN*.tfrecord.gz"]

                Paths can be local or GCS, the function will automatically care about downloading them if necessary.
            from_exam_converter: Whether the model uses data from the exam converter or not

        Returns: Generator yielding one feature_dict (= one sample) at a time
        """

        feature_batch_tensor = self._get_input_fn(path_globs, from_exam_converter)()

        with tf.Session() as sess:
            while True:
                try:
                    feature_dict_batch_np = sess.run(feature_batch_tensor)

                    batch_size = list(feature_dict_batch_np.values())[0].shape[0]
                    for sample_nr in range(batch_size):
                        feature_dict_np = {k: v[sample_nr] for k, v in iteritems(feature_dict_batch_np)}
                        yield feature_dict_np

                except tf.errors.OutOfRangeError:
                    break

    def feed(self):
        """
        Returns:
          images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """
        with tf.name_scope(self.name):
            feature_dicts = self.get_feature_dicts([self.tfrecords_file])
            images = []
            for dict in feature_dicts:
                image = tf.convert_to_tensor(dict[ec.FEATKEY_IMAGE])
                image = self.convert2float(image)
                images.append(image)

            # images = tf.stack(images)
            # images = self._preprocess(images)
            images = tf.train.shuffle_batch(
                [image], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=self.min_queue_examples,
                # allow_smaller_final_batch=True
            )
            tf.summary.image('input', images)

        return images

    def _preprocess(self, image):
        # image = tf.image.resize_images(image, size=(self.image_width, self.image_height))
        image = self.convert2float(image)

        return image


def test_reader():
    TRAIN_FILE_1 = 'data/tfrecords/apple.tfrecords'
    TRAIN_FILE_2 = 'data/tfrecords/orange.tfrecords'

    with tf.Graph().as_default():
        reader1 = Reader(TRAIN_FILE_1, batch_size=2)
        reader2 = Reader(TRAIN_FILE_2, batch_size=2)
        images_op1 = reader1.feed()
        images_op2 = reader2.feed()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                batch_images1, batch_images2 = sess.run([images_op1, images_op2])
                print("image shape: {}".format(batch_images1))
                print("image shape: {}".format(batch_images2))
                print("=" * 10)
                step += 1
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    test_reader()
