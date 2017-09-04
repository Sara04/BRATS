"""Class for BRATS 2017 scan/sample info loading and generating."""
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import morphology


class ScanBRATS(object):
    """Class for BRATS 2017 scan info loading and generating."""

    """
        Attributes:
            name: scan name
            relative_path: scan path relative to the database path
            mode: train, valid or test mode (dataset)

        Methods:
            compute_brain_mask: find binary mask that corresponds to the
                brain and skul region
            compute_tumor_distance_map: compute distances of healthy
                tissue voxels to tumorous tissue voxels
            load_volume: load a scan of chosen modality
            load_normalized_volume: load scan of a chosen modality with
                normalized values
            load_brain_mask: load brain and skul region mask
            load_tumor_distance_map: load distances to tumorous tissue
            load_volumes: load all modalities of a scan and
                segmentation, brain mask and tumor distance map optionally
    """
    def __init__(self, name, relative_path, mode):
        """Initialization of ScanBRATS attributes."""
        self.name = name
        self.relative_path = relative_path
        self.mode = mode

    def _compute_and_save_brain_mask(self, db, brain_mask_path):

        v_t1 = self.load_volume(db, 't1')
        v_t2 = self.load_volume(db, 't2')
        v_flair = self.load_volume(db, 'flair')
        v_t1ce = self.load_volume(db, 't1ce')

        bm = (v_t1 != 0) * (v_t2 != 0) * (v_flair != 0) * (v_t1ce != 0)
        bm.tofile(brain_mask_path)

    def compute_brain_mask(self, db, exp_out):
        """Compute brain mask."""
        """
            Arguments:
                db: DatabaseBRATS
                exp_out: experiment output for meta data
        """
        bm_path = os.path.join(exp_out, self.name + '_brain_mask.bin')
        self._compute_and_save_brain_mask(db, bm_path)

    def _compute_and_save_tumor_distance_map(self, db, brain_mask,
                                             tumor_distance_map_path):

        v_seg = self.load_volume(db, 'seg')
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
        tumor_dist_map.astype('uint8').tofile(tumor_distance_map_path)

    def compute_tumor_distance_map(self, db, bm_exp_out, tdm_exp_out):
        """Tumor distance map computation."""
        """
            Arguments:
                db: DatabaseBRATS
                exp_out: experiment output for meta data
        """
        tdm_path = os.path.join(tdm_exp_out, self.name + '_tumor_dist.bin')

        brain_mask = self.load_brain_mask(bm_exp_out)
        self._compute_and_save_tumor_distance_map(db, brain_mask, tdm_path)

    def load_volume(self, db, m):
        """Loading volume as numpy array."""
        """
            Arguments:
                db: DatabaseBRATS
                m: MRI modality
            Returns:
                volume as numpy array
        """
        volume_path = os.path.join(db.db_path, self.relative_path,
                                   self.name + '_' + m + '.nii')
        return nib.load(volume_path).get_data().astype('float32')

    def load_normalized_volume(self, db, m):
        """Loading normalized volume as numpy array."""
        """
            Arguments:
                db: DatabaseBRATS
                m: MRI modality
            Returns:
                volume as numpy array
        """
        volume_path = os.path.join(db.normalized_volumes_dir, self.name,
                                   self.name + '_' + m + '.bin')
        return np.reshape(np.fromfile(volume_path, dtype='float32'),
                          [db.h, db.w, db.d])

    def load_brain_mask(self, db):
        """Loading brain mask as numpy array."""
        """
            Arguments:
                db: DatabaseBRATS
            Returns:
                brain mask as a numpy array
        """
        brain_mask_path = os.path.join(db.brain_masks_dir,
                                       self.name + '_brain_mask.bin')
        np_array = np.fromfile(brain_mask_path, dtype='uint8')
        return np.reshape(np_array, (db.h, db.w, db.d))

    def load_tumor_dist_maps(self, db):
        """Loading tumor distance map as numpy array."""
        """
            Arguments:
                db: DatabaseBRATS
            Returns:
                tumor distance map as a numpy array
        """
        tdm_path = os.path.join(db.tumor_dist_dir,
                                self.name + '_tumor_dist.bin')
        np_array = np.fromfile(tdm_path, dtype='uint8')
        return np.reshape(np_array, (db.h, db.w, db.d))

    def load_volumes(self, db, load_normalized=False,
                     load_seg=True, load_bm=True, load_tdm=True):
        """Loading all volumes as a list numpy arrays."""
        """
            Arguments:
                db: DatabaseBRATS
                normalized: flag indicating whether to load normalized volumes
                load_seg: flag indicating whether to load segmentation
                load_bm: flag indicating whether to load brain mask
                load_tdm: flag indicating whether to load tumor distance map
            Returns:
                list of volumes
        """
        if load_normalized:
            volumes = [self.load_normalized_volume(db, m)
                       for m in db.modalities[:-1]]
        else:
            volumes = [self.load_volume(db, m) for m in db.modalities[:-1]]
        if load_seg:
            volumes.append(self.load_volume(db, 'seg'))
        if load_bm:
            volumes.append(self.load_brain_mask(db))
        if load_tdm:
            volumes.append(self.load_tumor_dist_maps(db))
        return volumes
