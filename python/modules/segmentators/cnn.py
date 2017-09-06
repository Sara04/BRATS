"""Segmentation class."""
import tensorflow as tf
import os
import cv2


def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def _bias_variable(shape):
    initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial)


def _conv2d(x, weights, strides=[1, 1, 1, 1], padding_mode='VALID'):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1],
                        padding=padding_mode)


def _max_pool_2x2(x, ksize=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='VALID')


class CnnBRATS(object):
    """Segmentation class."""

    def __init__(self, lp_w=33, lp_h=33, lp_d=11, sp_w=13, sp_h=13, sp_d=4):
        """Class initialization."""
        self.lp_w, self.lp_h, self.lp_d = [lp_w, lp_h, lp_d]
        self.sp_w, self.sp_h, self.sp_d = [sp_w, sp_h, sp_d]

        self.lp_x_r1 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.lp_h * self.lp_w * self.lp_d])
        self.sp_x_r1 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.sp_h * self.sp_w * self.sp_d])
        self.y_gt_r1 = tf.placeholder(tf.float32, shape=[None, 4])

        self.lp_x_r2 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.lp_h * self.lp_w * self.lp_d])
        self.sp_x_r2 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.sp_h * self.sp_w * self.sp_d])
        self.y_gt_r2 = tf.placeholder(tf.float32, shape=[None, 2])

        self.sess = tf.Session()

        lp_imgs_r1 = tf.reshape(self.lp_x_r1,
                                [-1, self.lp_h, self.lp_w, self.lp_d])
        sp_imgs_r1 = tf.reshape(self.sp_x_r1,
                                [-1, self.sp_h, self.sp_w, self.sp_d])

        lp_imgs_r2 = tf.reshape(self.lp_x_r2,
                                [-1, self.lp_h, self.lp_w, self.lp_d])
        sp_imgs_r2 = tf.reshape(self.sp_x_r2,
                                [-1, self.sp_h, self.sp_w, self.sp_d])

        with tf.variable_scope('l_patches'):
            with tf.variable_scope('layer_1'):
                lp_w_conv1 = _weight_variable([5, 5, 11, 32])
                lp_b_conv1 = _bias_variable([32])
                lp_h_conv1_r1 = tf.nn.relu(_conv2d(lp_imgs_r1, lp_w_conv1) +
                                           lp_b_conv1)
                lp_h_conv1_r2 = tf.nn.relu(_conv2d(lp_imgs_r2, lp_w_conv1) +
                                           lp_b_conv1)
            with tf.variable_scope('layer_2'):
                lp_w_conv2 = _weight_variable([5, 5, 32, 64])
                lp_b_conv2 = _bias_variable([64])
                lp_h_conv2_r1 = tf.nn.relu(_conv2d(lp_h_conv1_r1, lp_w_conv2) +
                                           lp_b_conv2)
                lp_h_pool2_r1 = _max_pool_2x2(lp_h_conv2_r1)
                lp_h_conv2_r2 = tf.nn.relu(_conv2d(lp_h_conv1_r2, lp_w_conv2) +
                                           lp_b_conv2)
                lp_h_pool2_r2 = _max_pool_2x2(lp_h_conv2_r2)
            with tf.variable_scope('layer_3'):
                lp_w_conv3 = _weight_variable([5, 5, 64, 128])
                lp_b_conv3 = _bias_variable([128])
                lp_h_conv3_r1 = tf.nn.relu(_conv2d(lp_h_pool2_r1, lp_w_conv3) +
                                           lp_b_conv3)
                lp_h_conv3_r2 = tf.nn.relu(_conv2d(lp_h_pool2_r2, lp_w_conv3) +
                                           lp_b_conv3)
                lp_h_conv3_flat_r1 =\
                    tf.reshape(lp_h_conv3_r1, [-1, 8 * 8 * 128])
                lp_h_conv3_flat_r2 =\
                    tf.reshape(lp_h_conv3_r2, [-1, 8 * 8 * 128])
            with tf.variable_scope('layer_4'):
                lp_w_fcn1 = _weight_variable([8 * 8 * 128, 512])
                lp_b_fcn1 = _bias_variable([512])
                lp_h_fcn1_r1 =\
                    tf.nn.relu(tf.matmul(lp_h_conv3_flat_r1, lp_w_fcn1) +
                               lp_b_fcn1)
                lp_h_fcn1_r2 =\
                    tf.nn.relu(tf.matmul(lp_h_conv3_flat_r2, lp_w_fcn1) +
                               lp_b_fcn1)
            with tf.variable_scope('layer_5'):
                lp_w_fcn2 = _weight_variable([512, 32])
                lp_b_fcn2 = _bias_variable([32])
                lp_h_fcn2_r1 = tf.nn.relu(tf.matmul(lp_h_fcn1_r1, lp_w_fcn2) +
                                          lp_b_fcn2)
                lp_h_fcn2_r2 = tf.nn.relu(tf.matmul(lp_h_fcn1_r2, lp_w_fcn2) +
                                          lp_b_fcn2)
        with tf.variable_scope('s_patches'):
            with tf.variable_scope('layer_1'):
                sp_w_conv1 = _weight_variable([5, 5, 4, 32])
                sp_b_conv1 = _bias_variable([32])
                sp_h_conv1_r1 = tf.nn.relu(_conv2d(sp_imgs_r1, sp_w_conv1) +
                                           sp_b_conv1)
                sp_h_conv1_r2 = tf.nn.relu(_conv2d(sp_imgs_r2, sp_w_conv1) +
                                           sp_b_conv1)
            with tf.variable_scope('layer_2'):
                sp_w_conv2 = _weight_variable([5, 5, 32, 64])
                sp_b_conv2 = _bias_variable([64])
                sp_h_conv2_r1 = tf.nn.relu(_conv2d(sp_h_conv1_r1, sp_w_conv2) +
                                           sp_b_conv2)
                sp_h_conv2_r2 = tf.nn.relu(_conv2d(sp_h_conv1_r2, sp_w_conv2) +
                                           sp_b_conv2)
                sp_h_conv2_flat_r1 =\
                    tf.reshape(sp_h_conv2_r1, [-1, 5 * 5 * 64])
                sp_h_conv2_flat_r2 =\
                    tf.reshape(sp_h_conv2_r2, [-1, 5 * 5 * 64])
            with tf.variable_scope('layer_3'):
                sp_w_fcn1 = _weight_variable([5 * 5 * 64, 256])
                sp_b_fcn1 = _bias_variable([256])
                sp_h_fcn1_r1 =\
                    tf.nn.relu(tf.matmul(sp_h_conv2_flat_r1, sp_w_fcn1) +
                               sp_b_fcn1)
                sp_h_fcn1_r2 =\
                    tf.nn.relu(tf.matmul(sp_h_conv2_flat_r2, sp_w_fcn1) +
                               sp_b_fcn1)
            with tf.variable_scope('layer_4'):
                sp_w_fcn2 = _weight_variable([256, 32])
                sp_b_fcn2 = _bias_variable([32])
                sp_h_fcn2_r1 = tf.nn.relu(tf.matmul(sp_h_fcn1_r1, sp_w_fcn2) +
                                          sp_b_fcn2)
                sp_h_fcn2_r2 = tf.nn.relu(tf.matmul(sp_h_fcn1_r2, sp_w_fcn2) +
                                          sp_b_fcn2)
        with tf.variable_scope('patch_merge'):
            with tf.variable_scope('layer_1'):
                feat_r1 = tf.concat([lp_h_fcn2_r1, sp_h_fcn2_r1], 1)
                feat_r2 = tf.concat([lp_h_fcn2_r2, sp_h_fcn2_r2], 1)
                mp_w_fcn1 = _weight_variable([64, 32])
                mp_b_fcn1 = _bias_variable([32])
                mp_h_fcn1_r1 = tf.nn.relu(tf.matmul(feat_r1, mp_w_fcn1) +
                                          mp_b_fcn1)
                mp_h_fcn1_r2 = tf.nn.relu(tf.matmul(feat_r2, mp_w_fcn1) +
                                          mp_b_fcn1)
            with tf.variable_scope('layer_2'):
                mp_w_fcn2 = _weight_variable([32, 4])
                mp_b_fcn2 = _bias_variable([4])
                mp_h_fcn1_r1 =\
                    tf.nn.softmax(tf.matmul(mp_h_fcn1_r1, mp_w_fcn2) +
                                  mp_b_fcn2)
                mp_h_fcn1_r2_4 =\
                    tf.nn.softmax(tf.matmul(mp_h_fcn1_r2, mp_w_fcn2) +
                                  mp_b_fcn2)
                mp_4_to_2 = tf.Variable([[1., 0], [0, 1.], [0, 1.], [0, 1.]])
                mp_h_fcn1_r2 = tf.matmul(mp_h_fcn1_r2_4, mp_4_to_2)

        # ___________________________________________________________________ #
        cross_entropy =\
            0.75 * tf.reduce_mean(-tf.reduce_sum(self.y_gt_r1 *
                                                 tf.log(mp_h_fcn1_r1),
                                                 reduction_indices=[1])) +\
            0.25 * tf.reduce_mean(-tf.reduce_sum(self.y_gt_r2 *
                                                 tf.log(mp_h_fcn1_r2),
                                                 reduction_indices=[1]))

        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
        correct_prediction_r1 = tf.equal(tf.argmax(mp_h_fcn1_r1, 1),
                                         tf.argmax(self.y_gt_r1, 1))
        self.accuracy_r1 =\
            tf.reduce_mean(tf.cast(correct_prediction_r1, tf.float32))
        self.results_r1 = mp_h_fcn1_r1

        correct_prediction_r2 = tf.equal(tf.argmax(mp_h_fcn1_r2, 1),
                                         tf.argmax(self.y_gt_r2, 1))
        self.accuracy_r2 =\
            tf.reduce_mean(tf.cast(correct_prediction_r2, tf.float32))
        self.results_r2 = mp_h_fcn1_r2

        vars_to_save = tf.trainable_variables()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(vars_to_save, max_to_keep=100000)

    def train(self, train_data_r1, train_data_r2):
        """Segmentation training."""
        self.sess.run(self.train_step,
                      feed_dict={self.lp_x_r1: train_data_r1['l_patch'],
                                 self.sp_x_r1: train_data_r1['s_patch'],
                                 self.y_gt_r1: train_data_r1['labels'],
                                 self.lp_x_r2: train_data_r2['l_patch'],
                                 self.sp_x_r2: train_data_r2['s_patch'],
                                 self.y_gt_r2: train_data_r2['labels']})

    def train_valid_accuracy(self,
                             train_data_r1, train_data_r2,
                             valid_data_r1, valid_data_r2):
        """Segmentation training and validation."""
        '''
        imgs_test = self.sess.run(self.x_image_p1,
                                   feed_dict={self.x_p1: training_data_p1,
                                              self.y_gt_p1: training_labels_p1,
                                              self.is_train: False})

        print("imgs test:", imgs_test.shape)
        for i in range(imgs_test.shape[0]):
            for j in range(imgs_test.shape[3]):
                img = imgs_test[i, :, :, j]
                cv2.imshow('img', img)
                cv2.waitKey(0)
        '''
        train_a_r1 = self.sess.run(self.accuracy_r1,
                                   feed_dict={self.lp_x_r1: train_data_r1['l_patch'],
                                              self.sp_x_r1: train_data_r1['s_patch'],
                                              self.y_gt_r1: train_data_r1['labels']})
        train_a_r2 = self.sess.run(self.accuracy_r2,
                                   feed_dict={self.lp_x_r2: train_data_r2['l_patch'],
                                              self.sp_x_r2: train_data_r2['s_patch'],
                                              self.y_gt_r2: train_data_r2['labels']})

        valid_a_r1 = self.sess.run(self.accuracy_r1,
                                   feed_dict={self.lp_x_r1: valid_data_r1['l_patch'],
                                              self.sp_x_r1: valid_data_r1['s_patch'],
                                              self.y_gt_r1: valid_data_r1['labels']})
        valid_a_r2 = self.sess.run(self.accuracy_r2,
                                   feed_dict={self.lp_x_r2: valid_data_r2['l_patch'],
                                              self.sp_x_r2: valid_data_r2['s_patch'],
                                              self.y_gt_r2: valid_data_r2['labels']})

        print("accuracy:" + " " + str(train_a_r1) + " " + str(train_a_r2) + " " + str(valid_a_r1) + " " + str(valid_a_r2))

    def test_accuracy(self, test_data):
        """Test segmentation."""
        test_labels = self.sess.run(self.results_r1,
                                    feed_dict={self.lp_x_r1: test_data['l_patch'],
                                               self.sp_x_r1: test_data['s_patch']})
        return test_labels

    def name(self):
        """Class name reproduction."""
        return "%s()" % (type(self).__name__)

    def restore_model(self, output_path, it):
        """Restore model."""
        model_path = os.path.join(output_path, self.name(),
                                  'model_' + str(it))
        self.saver.restore(self.sess, model_path)

    def save_model(self, output_path, it):
        """Saving model."""
        model_path = os.path.join(output_path, self.name(), 'model_' + str(it))

        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        self.saver.save(self.sess, model_path)
