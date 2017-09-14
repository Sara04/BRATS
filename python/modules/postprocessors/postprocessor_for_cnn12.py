"""Class for BRATS 2017 data postprocessing."""

import numpy as np
import json
import os
import cv2
import nibabel as nib
from scipy.ndimage import measurements
from scipy.ndimage import morphology

from .. import PostprocessorBRATS


class PostprocessorBRATSForCNN12(PostprocessorBRATS):
    """Class for BRATS 2017 data postprocessing."""

    def __init__(self, size_th_1=3000, size_th_2=1, score_th_1=0.915, score_th_2=1.0):
        """Initialization of PostprocessorBRATS attributes."""
        self.size_th_1 = size_th_1
        self.size_th_2 = size_th_2

        self.score_th_1 = score_th_1
        self.score_th_2 = score_th_2

    def _load_segmentation(self, db, scan, score=False, no=0):

        if not score:
            test_path = os.path.join(db.seg_results_dir, scan.name + '.bin')
        else:
            test_path = os.path.join(db.seg_results_dir, scan.name + '_scores.bin' + '_' + str(no))

        segmentation_test = np.reshape(np.fromfile(test_path),
                                       (scan.h, scan.w, scan.d))
        return segmentation_test

    def _compute_dice(self, a, b):

        sum_a = np.sum(a)
        sum_b = np.sum(b)
        sum_o = np.sum(a * b)

        return 2 * float(sum_o) / (sum_a + sum_b + 0.0001)

    def determine_parameters(self, db):

        dice_avg_whole = 0
        dice_avg_1 = 0
        dice_avg_2 = 0
        dice_avg_4 = 0
        count = 0
        for s_idx, scan_name in enumerate(db.train_valid):
            print("s idx:", s_idx)
            if s_idx > 32:
                continue
            segment_gt = db.train_dict[scan_name].load_volume(db, 'seg')
            segment_test = self._load_segmentation(db, db.train_dict[scan_name])
            segment_test_o = np.copy(segment_test)
            segment_sc_test_0 = self._load_segmentation(db, db.train_dict[scan_name], True, 0)
            segment_sc_test_1 = self._load_segmentation(db, db.train_dict[scan_name], True, 1)
            segment_sc_test_2 = self._load_segmentation(db, db.train_dict[scan_name], True, 2)
            segment_sc_test_4 = self._load_segmentation(db, db.train_dict[scan_name], True, 4)
            segment_sc_test_t = segment_sc_test_1 + segment_sc_test_2 + segment_sc_test_4
            segment_sc_test_m = np.maximum(np.maximum(segment_sc_test_1, segment_sc_test_2), segment_sc_test_4)

            mask_whole = (segment_test != 0) * (segment_sc_test_t > self.score_th_1) * (segment_sc_test_m > 0.4)
            M, label = measurements.label(mask_whole)

            for i in range(1, label + 1):
                p = (M == i)
                p_sum = np.sum(p)
                p_sc = (M == i) * segment_sc_test_t
                p_sc_sum = np.sum(p_sc)
                p_avg = p_sc_sum / p_sum
                #p_1 = float(np.sum(p * (segment_test == 1))) / p_sum
                #p_2 = float(np.sum(p * (segment_test == 2))) / p_sum
                #p_4 = float(np.sum(p * (segment_test == 4))) / p_sum
                #print("p_sum, p_avg:", p_sum, p_avg)
                if p_sum < self.size_th_1:
                    mask_whole[p] = 0
                #if p_1 > 0.9999 or p_2 > 0.9999 or p_4 > 0.9999:
                #    mask_whole[p] = 0
                '''
                else:
                    print("p1, p2, p4:", p_1, p_2, p_4)
                    print("p_sum, p_avg:", p_sum, p_avg)
                    print("\n")
                '''
            se = np.ones((3, 3, 3))
            mask_whole = morphology.binary_closing(mask_whole, se)
            segment_test *= mask_whole

            dice_whole = self._compute_dice(segment_test != 0, segment_gt != 0)
            dice_1 = self._compute_dice(segment_test == 1, segment_gt == 1)
            dice_2 = self._compute_dice(segment_test == 2, segment_gt == 2)
            dice_4 = self._compute_dice(segment_test == 4, segment_gt == 4)

            dice_whole_o = self._compute_dice(segment_test_o != 0, segment_gt != 0)
            dice_1_o = self._compute_dice(segment_test_o == 1, segment_gt == 1)
            dice_2_o = self._compute_dice(segment_test_o == 2, segment_gt == 2)
            dice_4_o = self._compute_dice(segment_test_o == 4, segment_gt == 4)

            print("dice whole:", dice_whole, dice_whole_o)
            print("dice 1:", dice_1, dice_1_o)
            print("dice 2:", dice_2, dice_2_o)
            print("dice 4:", dice_4, dice_4_o)
            print("\n")
            if dice_whole != 0:
                dice_avg_whole += dice_whole
                dice_avg_1 += dice_1
                dice_avg_2 += dice_2
                dice_avg_4 += dice_4
                count += 1

        print("dice avg whole:", dice_avg_whole / count)
        print("dice avg 1:", dice_avg_1 / count)
        print("dice avg 2:", dice_avg_2 / count)
        print("dice avg 4:", dice_avg_4 / count)

    def postprocess(self, db):

        for s_idx, scan_name in enumerate(db.valid_dict.keys()):
            print("s idx:", s_idx)

            segment_test = self._load_segmentation(db, db.valid_dict[scan_name])
            segment_sc_test = self._load_segmentation(db, db.valid_dict[scan_name], True)
            #segment_sc_test_0 = self._load_segmentation(db, db.valid_dict[scan_name], True, 0)
            segment_sc_test_1 = self._load_segmentation(db, db.valid_dict[scan_name], True, 1)
            segment_sc_test_2 = self._load_segmentation(db, db.valid_dict[scan_name], True, 2)
            segment_sc_test_4 = self._load_segmentation(db, db.valid_dict[scan_name], True, 4)
            segment_sc_test_t = segment_sc_test_1 + segment_sc_test_2 + segment_sc_test_4
            segment_sc_test_m = np.maximum(np.maximum(segment_sc_test_1, segment_sc_test_2), segment_sc_test_4)

            mask_whole = (segment_test != 0) * (segment_sc_test_t > self.score_th_1) * (segment_sc_test_m > 0.4)
            M, label = measurements.label(mask_whole)

            for i in range(1, label + 1):
                p = (M == i)
                p_sum = np.sum(p)
                p_sc = (M == i) * segment_sc_test
                p_sc_sum = np.sum(p_sc)
                p_avg = p_sc_sum / p_sum
                #print("p_sum, p_avg:", p_sum, p_avg)
                if p_sum < self.size_th_1:
                    mask_whole[p] = 0

            se = np.ones((3, 3, 3))
            mask_whole = morphology.binary_closing(mask_whole, se)
            segment_test *= mask_whole

            segment_test.astype('int16')

            segment_nib = nib.Nifti1Image(segment_test, np.eye(4))

            output_path = os.path.join(db.seg_results_final_dir, scan_name + '.nii.gz')
            segment_nib.to_filename(output_path)

    def name(self):
        """Class name reproduction."""
        return "%s()" % (type(self).__name__)
