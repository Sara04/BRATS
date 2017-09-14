"""Segmentation class."""
import tensorflow as tf
import os
import numpy as np

from .. import SegmentatorBRATS
from cnn_utils import _weight_variable_xavier, _bias_variable
from cnn_utils import _conv, _max_pool


class CnnBRATS2(SegmentatorBRATS):
    """Segmentation class CnnBRATS2."""

    """
        Arguments:
            lr: learning rate
            lw: two and four classification problems loss weights
            kp: keep probability for drop out regularization
            restore: flag indicating whether to restore model
            restore_it: train iteration of the model to be restored
            lp_w, lp_h, lp_d: large patch width, height and depth
                (8 volume patches and 3 distance maps patches)
            sp_w, sp_h, sp_d: small patch width, height and depth
                (4 volume patches)

        Methods:
            training_and_validation: training and validation of
                algorithm on training dataset
            compute_classification_scores: computation of
                classification scores
            save_model: saving trained model
            restore_model: restoreing trained model
    """
    def __init__(self, lr=10e-4, lw=[0.25, 0.75], kp=0.5,
                 restore=False, restore_it=0,
                 train_iters=100000,
                 lp_w=45, lp_h=45, lp_d=11, sp_w=17, sp_h=17, sp_d=4):
        """Class initialization."""
        self.lr, self.lw, self.kp = [lr, lw, kp]

        self.restore, self.restore_it = [restore, restore_it]

        self.train_iters = train_iters

        self.keep_prob = tf.placeholder(tf.float32)
        self.lp_w, self.lp_h, self.lp_d = [lp_w, lp_h, lp_d]
        self.sp_w, self.sp_h, self.sp_d = [sp_w, sp_h, sp_d]

        self.lp_x_r1 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.lp_h * self.lp_w * self.lp_d])
        self.sp_x_r1 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.sp_h * self.sp_w * self.sp_d])
        self.gt_r1 = tf.placeholder(tf.float32, shape=[None, 4])

        self.lp_x_r2 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.lp_h * self.lp_w * self.lp_d])
        self.sp_x_r2 =\
            tf.placeholder(tf.float32,
                           shape=[None, self.sp_h * self.sp_w * self.sp_d])
        self.gt_r2 = tf.placeholder(tf.float32, shape=[None, 2])

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
                lp_w_c1 = _weight_variable_xavier([5, 5, 11, 32])
                lp_b_c1 = _bias_variable([32])
                lp_h_c1_r1 = tf.nn.relu(_conv(lp_imgs_r1, lp_w_c1) + lp_b_c1)
                lp_h_c1_r2 = tf.nn.relu(_conv(lp_imgs_r2, lp_w_c1) + lp_b_c1)
                lp_h_p1_r1 = _max_pool(lp_h_c1_r1)
                lp_h_p1_r2 = _max_pool(lp_h_c1_r2)
            with tf.variable_scope('layer_2'):
                lp_w_c2 = _weight_variable_xavier([5, 5, 32, 64])
                lp_b_c2 = _bias_variable([64])
                lp_h_c2_r1 = tf.nn.relu(_conv(lp_h_p1_r1, lp_w_c2) + lp_b_c2)
                lp_h_c2_r2 = tf.nn.relu(_conv(lp_h_p1_r2, lp_w_c2) + lp_b_c2)
                lp_h_p2_r1 = _max_pool(lp_h_c2_r1)
                lp_h_p2_r2 = _max_pool(lp_h_c2_r2)
            with tf.variable_scope('layer_3'):
                lp_w_c3 = _weight_variable_xavier([5, 5, 64, 128])
                lp_b_c3 = _bias_variable([128])
                lp_h_c3_r1 = tf.nn.relu(_conv(lp_h_p2_r1, lp_w_c3) + lp_b_c3)
                lp_h_c3_r2 = tf.nn.relu(_conv(lp_h_p2_r2, lp_w_c3) + lp_b_c3)
                lp_h_c3_fl_r1 = tf.reshape(lp_h_c3_r1, [-1, 4 * 4 * 128])
                lp_h_c3_fl_r2 = tf.reshape(lp_h_c3_r2, [-1, 4 * 4 * 128])
            with tf.variable_scope('layer_4'):
                lp_w_fcn1 = _weight_variable_xavier([4 * 4 * 128, 128])
                lp_b_fcn1 = _bias_variable([128])
                lp_h_fcn1_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(lp_h_c3_fl_r1,
                                                       lp_w_fcn1) +
                                             lp_b_fcn1),
                                  self.keep_prob)
                lp_h_fcn1_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(lp_h_c3_fl_r2,
                                                       lp_w_fcn1) +
                                             lp_b_fcn1),
                                  self.keep_prob)
            with tf.variable_scope('layer_5'):
                lp_w_fcn2 = _weight_variable_xavier([128, 32])
                lp_b_fcn2 = _bias_variable([32])
                lp_h_fcn2_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(lp_h_fcn1_r1,
                                                       lp_w_fcn2) +
                                             lp_b_fcn2),
                                  self.keep_prob)
                lp_h_fcn2_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(lp_h_fcn1_r2,
                                                       lp_w_fcn2) +
                                             lp_b_fcn2),
                                  self.keep_prob)

        with tf.variable_scope('s_patches'):
            with tf.variable_scope('layer_1'):
                sp_w_c1 = _weight_variable_xavier([5, 5, 4, 16])
                sp_b_c1 = _bias_variable([16])
                sp_h_c1_r1 = tf.nn.relu(_conv(sp_imgs_r1, sp_w_c1) + sp_b_c1)
                sp_h_c1_r2 = tf.nn.relu(_conv(sp_imgs_r2, sp_w_c1) + sp_b_c1)
            with tf.variable_scope('layer_2'):
                sp_w_c2 = _weight_variable_xavier([5, 5, 16, 32])
                sp_b_c2 = _bias_variable([32])
                sp_h_c2_r1 = tf.nn.relu(_conv(sp_h_c1_r1, sp_w_c2) + sp_b_c2)
                sp_h_c2_r2 = tf.nn.relu(_conv(sp_h_c1_r2, sp_w_c2) + sp_b_c2)
            with tf.variable_scope('layer_3'):
                sp_w_c3 = _weight_variable_xavier([5, 5, 32, 64])
                sp_b_c3 = _bias_variable([64])
                sp_h_c3_r1 = tf.nn.relu(_conv(sp_h_c2_r1, sp_w_c3) + sp_b_c3)
                sp_h_c3_r2 = tf.nn.relu(_conv(sp_h_c2_r2, sp_w_c3) + sp_b_c3)
                sp_h_c3_fl_r1 = tf.reshape(sp_h_c3_r1, [-1, 5 * 5 * 64])
                sp_h_c3_fl_r2 = tf.reshape(sp_h_c3_r2, [-1, 5 * 5 * 64])
            with tf.variable_scope('layer_4'):
                sp_w_fcn1 = _weight_variable_xavier([5 * 5 * 64, 128])
                sp_b_fcn1 = _bias_variable([128])
                sp_h_fcn1_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(sp_h_c3_fl_r1,
                                                       sp_w_fcn1) +
                                             sp_b_fcn1),
                                  self.keep_prob)
                sp_h_fcn1_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(sp_h_c3_fl_r2,
                                                       sp_w_fcn1) +
                                             sp_b_fcn1),
                                  self.keep_prob)
            with tf.variable_scope('layer_5'):
                sp_w_fcn2 = _weight_variable_xavier([128, 32])
                sp_b_fcn2 = _bias_variable([32])
                sp_h_fcn2_r1 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(sp_h_fcn1_r1,
                                                       sp_w_fcn2) +
                                             sp_b_fcn2),
                                  self.keep_prob)
                sp_h_fcn2_r2 =\
                    tf.nn.dropout(tf.nn.relu(tf.matmul(sp_h_fcn1_r2,
                                                       sp_w_fcn2) +
                                             sp_b_fcn2),
                                  self.keep_prob)

        with tf.variable_scope('patch_merge'):
            with tf.variable_scope('layer_1'):
                feat_r1 = tf.concat([lp_h_fcn2_r1, sp_h_fcn2_r1], 1)
                feat_r2 = tf.concat([lp_h_fcn2_r2, sp_h_fcn2_r2], 1)
                mp_w_fcn1 = _weight_variable_xavier([64, 32])
                mp_b_fcn1 = _bias_variable([32])
                mp_h_fcn1_r1 = tf.nn.relu(tf.matmul(feat_r1, mp_w_fcn1) +
                                          mp_b_fcn1)
                mp_h_fcn1_r2 = tf.nn.relu(tf.matmul(feat_r2, mp_w_fcn1) +
                                          mp_b_fcn1)
            with tf.variable_scope('layer_2'):
                mp_w_fcn2 = _weight_variable_xavier([32, 4])
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
            self.lw[0] *\
            tf.reduce_mean(-tf.reduce_sum(self.gt_r1 * tf.log(mp_h_fcn1_r1),
                                          reduction_indices=[1])) +\
            self.lw[1] *\
            tf.reduce_mean(-tf.reduce_sum(self.gt_r2 * tf.log(mp_h_fcn1_r2),
                                          reduction_indices=[1]))
        self.train_step =\
            tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)

        clf_r1 = tf.equal(tf.argmax(mp_h_fcn1_r1, 1), tf.argmax(self.gt_r1, 1))
        self.accuracy_r1 = tf.reduce_mean(tf.cast(clf_r1, tf.float32))
        self.probabilities_1 = mp_h_fcn1_r2_4

        clf_r2 = tf.equal(tf.argmax(mp_h_fcn1_r2, 1), tf.argmax(self.gt_r2, 1))
        self.accuracy_r2 = tf.reduce_mean(tf.cast(clf_r2, tf.float32))
        self.probabilities_2 = mp_4_to_2

        vars_to_save = tf.trainable_variables()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(vars_to_save, max_to_keep=100000)

    def training_and_validation(self, db, prep, patch_ex, exp_out):
        """Run slgorithm's training and validation."""
        """
            Arguments:
                db: DatabaseBRATS object
                prep: PreprocessorBRATS object
                patch_ex: PatchExtractorBRATS object
                exp_out: path to the experiment output
        """
        seg_path = os.path.join(exp_out, 'segmentators', db.name(),
                                prep.name(), patch_ex.name(), self.name())
        seg_done = os.path.join(seg_path, 'done')

        if not os.path.exists(seg_done):
            if self.restore:
                self.restore_model(seg_path, self.restore_it)

            if not os.path.exists(seg_path):
                os.makedirs(seg_path)

            for it in range(self.restore_it, self.train_iters):

                self._train(db, prep, patch_ex, exp_out)

                if it % 10 == 0:

                    train_acc_l_region, train_acc_s_region =\
                        self._validate(db, prep, patch_ex, exp_out, 'train')
                    valid_acc_l_region, valid_acc_s_region =\
                        self._validate(db, prep, patch_ex, exp_out, 'valid')

                    print("train, valid accuracy:" + " " +
                          str(train_acc_l_region) + " " +
                          str(train_acc_s_region) + " " +
                          str(valid_acc_l_region) + " " +
                          str(valid_acc_s_region))
                if it % 50 == 0:
                    self.save_model(seg_path, it)
        else:
            print "Segmentator is already trained!"

    def _train(self, db, prep, patch_ex, exp_out):
        data = patch_ex.extract_train_or_valid_data(db, prep, exp_out, 'train')

        if data['region_1']['labels'].shape[0]:
            random_s =\
                np.random.choice(data['region_1']['labels'].shape[0],
                                 int(0.4 * data['region_1']['labels'].shape[0]),
                                 replace=False)
            for idx, r in enumerate(random_s):
                data['region_1']['labels'][r, :] = np.zeros(4)
                data['region_1']['labels'][r, np.random.randint(4)] = 1

            random_s =\
                np.random.choice(data['region_2']['labels'].shape[0],
                                 int(0.2 * data['region_2']['labels'].shape[0]),
                                 replace=False)
            for idx, r in enumerate(random_s):
                data['region_2']['labels'][r, :] = np.zeros(2)
                data['region_2']['labels'][r, np.random.randint(2)] = 1

            self.sess.run(self.train_step,
                          feed_dict={self.lp_x_r1: data['region_1']['l_patch'],
                                     self.sp_x_r1: data['region_1']['s_patch'],
                                     self.gt_r1: data['region_1']['labels'],
                                     self.lp_x_r2: data['region_2']['l_patch'],
                                     self.sp_x_r2: data['region_2']['s_patch'],
                                     self.gt_r2: data['region_2']['labels'],
                                     self.keep_prob: self.kp})

    def _validate(self, db, prep, patch_ex, exp_out, subset):
        if subset == 'train':
            data = patch_ex.extract_train_or_valid_data(db, prep, exp_out, 'train_valid')
        elif subset == 'valid':
            data = patch_ex.extract_train_or_valid_data(db, prep, exp_out, 'valid')

        accuracy_l_region =\
            self.sess.run(self.accuracy_r1,
                          feed_dict={self.lp_x_r1: data['region_1']['l_patch'],
                                     self.sp_x_r1: data['region_1']['s_patch'],
                                     self.gt_r1: data['region_1']['labels'],
                                     self.keep_prob: 1.0})
        accuracy_s_region =\
            self.sess.run(self.accuracy_r2,
                          feed_dict={self.lp_x_r2: data['region_2']['l_patch'],
                                     self.sp_x_r2: data['region_2']['s_patch'],
                                     self.gt_r2: data['region_2']['labels'],
                                     self.keep_prob: 1.0})

        return accuracy_l_region, accuracy_s_region

    def _compute_clf_scores_per_scan(self, db, prep, patch_ex, clf_out, scan):

        scan_class_path = os.path.join(clf_out, scan.name + '.bin')
        scan_prob_path = os.path.join(clf_out, scan.name + '_scores.bin')
        volumes = scan.load_volumes(db, load_normalized=True)

        indices = np.where(volumes[5])
        class_number = np.zeros((scan.h, scan.w, scan.d))
        p_0 = np.zeros((scan.h, scan.w, scan.d))
        p_1 = np.zeros((scan.h, scan.w, scan.d))
        p_2 = np.zeros((scan.h, scan.w, scan.d))
        p_4 = np.zeros((scan.h, scan.w, scan.d))

        n_indices = len(indices[0])
        i = 0
        while i < n_indices:
            s_idx = [indices[0][i:i + patch_ex.test_patches_per_scan],
                     indices[1][i:i + patch_ex.test_patches_per_scan],
                     indices[2][i:i + patch_ex.test_patches_per_scan]]
            i += patch_ex.test_patches_per_scan
            patches = patch_ex.extract_test_patches(scan, db, prep,
                                                    volumes, s_idx)
            labels =\
                self.sess.run(self.probabilities_1,
                              feed_dict={self.lp_x_r1: patches['l_patch'],
                                         self.sp_x_r1: patches['s_patch'],
                                         self.keep_prob: 1.0})
            for j in range(labels.shape[0]):
                class_number[s_idx[0][j], s_idx[1][j], s_idx[2][j]] =\
                    db.classes[np.argmax(labels[j, :])]
                p_0[s_idx[0][j], s_idx[1][j], s_idx[2][j]] = labels[j, 0]
                p_1[s_idx[0][j], s_idx[1][j], s_idx[2][j]] = labels[j, 1]
                p_2[s_idx[0][j], s_idx[1][j], s_idx[2][j]] = labels[j, 2]
                p_4[s_idx[0][j], s_idx[1][j], s_idx[2][j]] = labels[j, 3]
        class_number.tofile(scan_class_path)
        p_0.tofile(scan_prob_path + '_0')
        p_1.tofile(scan_prob_path + '_1')
        p_2.tofile(scan_prob_path + '_2')
        p_4.tofile(scan_prob_path + '_4')

    def restore_model(self, input_path, it):
        """Restoring trained segmentation model."""
        """
            Arguments:
                input_path: path to the input directory
                it: train iteration of the model to be restored
        """
        model_path = os.path.join(input_path, self.name(), 'model_' + str(it))
        self.saver.restore(self.sess, model_path)

    def save_model(self, output_path, it):
        """Saving trained segmentation model."""
        """
            Arguments:
                output_path: path to the output directory
                it:  train iteration of the model to be saved
        """
        model_path = os.path.join(output_path, self.name(), 'model_' + str(it))
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        self.saver.save(self.sess, model_path)

    def name(self):
        """Class name reproduction."""
        """
            Returns segmentator's name.
        """
        return ("%s(lp_w=%s, lp_h=%s, lp_d=%s, sp_w=%s, sp_h=%s, sp_d=%s)"
                % (type(self).__name__,
                   self.lp_w, self.lp_h, self.lp_d,
                   self.sp_w, self.sp_h, self.sp_d))
