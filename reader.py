from __future__ import absolute_import, division, unicode_literals

import multiprocessing as mp

import tensorflow as tf

from common.train import inputs
from common.util import file_util, mx_logging
import numpy as np
import projects.edison.constants as ec
from projects.edison.train.model_preprocessors.exam_image_splitting import ExamToImagesSplitterModelPreprocessor
# import utils.py


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

    def feed(self):
        """
        Returns:
          images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """
        with tf.name_scope(self.name):
            schema = {
                ec.FEATKEY_NR_IMAGES: tf.io.FixedLenFeature(dtype=tf.int64, shape=[]),
                ec.FEATKEY_IMAGE: tf.io.FixedLenFeature(dtype=tf.string, shape=[]),
                ec.FEATKEY_IMAGE_HEIGHT: tf.io.FixedLenFeature(dtype=tf.int64, shape=[]),
                ec.FEATKEY_IMAGE_WIDTH: tf.io.FixedLenFeature(dtype=tf.int64, shape=[]),
                # NOTE: For some reason, "allow_missing" has to be set to True, otherwise there's an exception
                #       "allow_missing must be set to True". This has no relevance for us, our values are never missing.
                ec.FEATKEY_ORIGINAL_HEIGHT: tf.io.FixedLenSequenceFeature(dtype=tf.int64, shape=[], allow_missing=True),
                ec.FEATKEY_ORIGINAL_WIDTH: tf.io.FixedLenSequenceFeature(dtype=tf.int64, shape=[], allow_missing=True),
                ec.LABELKEY_GT_ROIS_WITH_FINDING_CODES: tf.io.FixedLenSequenceFeature(
                    dtype=tf.int64, shape=[100, 5], allow_missing=True
                ),
                ec.LABELKEY_GT_ANNOTATION_CONFIDENCES: tf.io.FixedLenSequenceFeature(
                    dtype=tf.int64, shape=[100], allow_missing=True
                ),
                ec.LABELKEY_GT_BIOPSY_PROVEN: tf.io.FixedLenSequenceFeature(
                    dtype=np.int64, shape=[100], allow_missing=True
                ),
                ec.LABELKEY_GT_BIRADS_SCORE: tf.io.FixedLenSequenceFeature(
                    dtype=np.int64, shape=[100], allow_missing=True
                ),
                ec.FEATKEY_ANNOTATION_STATUS: tf.io.FixedLenFeature(dtype=tf.string, shape=[]),
                ec.LABELKEY_WAS_REFERRED: tf.io.FixedLenFeature(dtype=tf.int64, shape=[]),
                ec.LABELKEY_WAS_REFERRED_IN_FOLLOWUP: tf.io.FixedLenFeature(dtype=tf.int64, shape=[]),
                ec.LABELKEY_BIOPSY_SCORE: tf.io.FixedLenFeature(dtype=tf.int64, shape=[]),
                ec.LABELKEY_BIOPSY_SCORE_OF_FOLLOWUP: tf.io.FixedLenFeature(dtype=tf.int64, shape=[]),
                ec.FEATKEY_STUDY_INSTANCE_UID: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                ec.LABELKEY_ANNOTATION_ID: tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
                ec.LABELKEY_ANNOTATOR_MAIL: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                ec.FEATKEY_SOP_INSTANCE_UID: tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string,
                                                                           allow_missing=True),
                ec.FEATKEY_PATIENT_ID: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                ec.FEATKEY_MANUFACTURER: tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                ec.FEATKEY_LATERALITY: tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string, allow_missing=True),
                ec.FEATKEY_VIEW: tf.io.FixedLenSequenceFeature(shape=[], dtype=tf.string, allow_missing=True),
                ec.LABELKEY_DENSITY: tf.io.FixedLenFeature(dtype=tf.int64, shape=[]),
                ec.LABELKEY_INTERVAL_TYPE: tf.io.FixedLenFeature(dtype=tf.int64, shape=[]),
            }
            tfrecord_paths = self.tfrecords_file

            mx_logging.info("Constructing train input fn for [{}, ...]".format(tfrecord_paths[0]))

            input_fn = inputs.input_fn_factory(
                tfrecord_paths,
                schema,
                self.batch_size,
                model_preprocessors=[ExamToImagesSplitterModelPreprocessor(self.output_dim)],
                num_epochs=None,
                num_parallel_reads=mp.cpu_count(),
                num_parallel_calls=mp.cpu_count(),
            )

            features, labels = input_fn()
            images = features[ec.FEATKEY_IMAGE]

        return images

    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_width, self.image_height))
        image = utils.convert2float(image)
        image.set_shape([self.image_width, self.image_height, 3])
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
