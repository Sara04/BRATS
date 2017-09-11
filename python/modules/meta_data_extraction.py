"""Class for BRATS 2017 meta data extraction."""
import os
import sys
import numpy as np
from scipy.ndimage import morphology


class MetaDataExtractorBRATS(object):
    """Class for BRATS 2017 database management."""

    """
        Methods:
            compute_brain_masks: compute and save brain masks
            compute_tumor_distance_maps: compute and save
                tumor distance maps
    """

    def _compute_and_save_brain_mask(self, scan, db, brain_mask_path):

        bm_path = os.path.join(db.brain_masks_dir,
                               scan.name + '_brain_mask.bin')
        v_t1 = scan.load_volume(db, 't1')
        v_t2 = scan.load_volume(db, 't2')
        v_flair = scan.load_volume(db, 'flair')
        v_t1ce = scan.load_volume(db, 't1ce')

        bm = (v_t1 != 0) * (v_t2 != 0) * (v_flair != 0) * (v_t1ce != 0)
        bm.tofile(bm_path)

    def compute_brain_masks(self, db, exp_out, mode):
        """Compute and save brain masks."""
        """
            Arguments:
                db: DatabaseBRATS object
                exp_out: path to the experiment meta output
                mode: training or validation subsets
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'valid':
            data_dict = db.valid_dict
        elif mode == 'test':
            data_dict = db.test_dict

        bm_output_path = os.path.join(exp_out, 'brain_masks', mode)
        db.brain_masks_dir = bm_output_path
        done_path = os.path.join(bm_output_path, 'done')
        if not os.path.exists(done_path):
            n_subjects = len(data_dict)
            if not os.path.exists(bm_output_path):
                os.makedirs(bm_output_path)
            for s_idx, s in enumerate(data_dict):
                self._compute_and_save_brain_mask(data_dict[s], db)
                sys.stdout.write("\rComputing and saving brain masks: "
                                 "%.3f %% / 100 %%" %
                                 (100 * float(s_idx + 1) / n_subjects))
                sys.stdout.flush()
            sys.stdout.write("\n")
            with open(done_path, 'w') as f:
                f.close()
        else:
            print "Brain masks already computed"

    def _compute_and_save_tumor_distance_map(self, scan, db):

        tdm_path = os.path.join(db.tumor_dist_dir,
                                scan.name + '_tumor_dist.bin')

        brain_mask = scan.load_brain_mask(db)
        v_seg = scan.load_volume(db, 'seg')
        seg_mask = v_seg != 0
        struct_elem = np.ones((3, 3, 3))
        v = 1.0

        tumor_dist_map = np.zeros((db.h, db.w, db.d))
        while(np.sum(seg_mask) != db.w * db.h * db.d):
            seg_mask_d = morphology.binary_dilation(seg_mask, struct_elem)
            tumor_dist_map += (seg_mask_d - seg_mask) * v
            seg_mask = np.copy(seg_mask_d)
            v += 1

        tumor_dist_map *= brain_mask
        tumor_dist_map = np.clip(tumor_dist_map, a_min=0.0, a_max=255.0)
        tumor_dist_map.astype('uint8').tofile(tdm_path)

    def compute_tumor_distance_maps(self, db, exp_out):
        """Compute and save tumor distance maps."""
        """
            Arguments:
                db: DatabaseBRATS object
                exp_out: path to the experiment meta output
        """
        tdm_output_path = os.path.join(exp_out, 'tumor_dist_maps', 'train')
        db.tumor_dist_dir = tdm_output_path
        done_path = os.path.join(tdm_output_path, 'done')
        if not os.path.exists(done_path):
            n_subjects = len(db.train_dict)
            if not os.path.exists(tdm_output_path):
                os.makedirs(tdm_output_path)
            for s_idx, s in enumerate(db.train_dict):
                self._compute_and_save_tumor_distance_map(db.train_dict[s], db)
                sys.stdout.write("\rComputing and saving tumor distance maps: "
                                 "%.3f %% / 100 %%" %
                                 (100 * float(s_idx + 1) / n_subjects))
                sys.stdout.flush()
            sys.stdout.write("\n")
            with open(done_path, 'w') as f:
                f.close()
        else:
            print "Tumor distance maps already computed"
