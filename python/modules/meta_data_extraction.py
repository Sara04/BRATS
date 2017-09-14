"""Class for BRATS 2017 meta data extraction."""
import os
import sys
import numpy as np
from scipy.ndimage import morphology


class MetaDataExtractorBRATS(object):
    """Class for BRATS 2017 meta data extraction."""

    """
        Methods:
            compute_brain_masks: compute and save brain masks
            compute_tumor_distance_maps: compute and save
                tumor distance maps
            compute_normalized_volumes: compute and save
                normalized volumes
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
                mode: train, valid or test database
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'valid':
            data_dict = db.valid_dict
        elif mode == 'test':
            data_dict = db.test_dict

        db.brain_masks_dir = os.path.join(exp_out, 'brain_masks', mode)
        if not os.path.exists(os.path.join(db.brain_masks_dir, 'done')):
            n_subjects = len(data_dict)
            if not os.path.exists(db.brain_masks_dir):
                os.makedirs(db.brain_masks_dir)
            for s_idx, s in enumerate(data_dict):
                self._compute_and_save_brain_mask(data_dict[s], db)
                sys.stdout.write("\rComputing and saving brain masks: "
                                 "%.3f %% / 100 %%" %
                                 (100 * float(s_idx + 1) / n_subjects))
                sys.stdout.flush()
            sys.stdout.write("\n")
            with open(os.path.join(db.brain_masks_dir, 'done'), 'w') as f:
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
        db.tumor_dist_dir = os.path.join(exp_out, 'tumor_dist_maps', 'train')
        if not os.path.exists(os.path.join(db.tumor_dist_dir, 'done')):
            n_subjects = len(db.train_dict)
            if not os.path.exists(db.tumor_dist_dir):
                os.makedirs(db.tumor_dist_dir)
            for s_idx, s in enumerate(db.train_dict):
                self._compute_and_save_tumor_distance_map(db.train_dict[s], db)
                sys.stdout.write("\rComputing and saving tumor distance maps: "
                                 "%.3f %% / 100 %%" %
                                 (100 * float(s_idx + 1) / n_subjects))
                sys.stdout.flush()
            sys.stdout.write("\n")
            with open(os.path.join(db.tumor_dist_dir, 'done'), 'w') as f:
                f.close()
        else:
            print "Tumor distance maps already computed"

    def _normalize_volumes(self, scan, db, prep):

        n_volumes = prep.normalize_volumes(db, scan)
        for m in db.modalities:
            n_volume_path = os.path.join(db.norm_volumes_dir,
                                         scan.name,
                                         scan.name + '_' + m + '.bin')
            if not os.path.exists(os.path.dirname(n_volume_path)):
                os.makedirs(os.path.dirname(n_volume_path))
            n_volumes[m].tofile(n_volume_path)

    def compute_normalized_volumes(self, db, prep, exp_out, mode):
        """Compute and save normalized volumes."""
        """
            Arguments:
                db: DatabaseBRATS object
                prep: PreprocessorBRATS object
                exp_out: path to the experiment meta data output
                mode: train, valid or test database
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'valid':
            data_dict = db.valid_dict
        elif mode == 'test':
            data_dict = db.test_dict

        db.norm_volumes_dir = os.path.join(exp_out,
                                           'normalized_volumes', mode)
        if not os.path.exists(os.path.join(db.norm_volumes_dir, 'done')):
            n_subjects = len(data_dict)
            if not os.path.exists(db.norm_volumes_dir):
                os.makedirs(db.norm_volumes_dir)
            for s_idx, s in enumerate(data_dict):
                self._normalize_volumes(data_dict[s], db, prep)
                sys.stdout.write("\rComputing and saving normalized volumes: "
                                 "%.3f %% / 100 %%" %
                                 (100 * float(s_idx + 1) / n_subjects))
                sys.stdout.flush()
            sys.stdout.write("\n")

            with open(os.path.join(db.norm_volumes_dir, 'done'), 'w') as f:
                f.close()
        else:
            print "Volumes already normalized"
