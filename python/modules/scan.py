"""Class for BRATS 2017 scan/sample info loading and generating."""
import os
import nibabel as nib
import numpy as np


class ScanBRATS(object):
    """Class for BRATS 2017 scan info loading and generating."""

    """
        Attributes:
            name: scan name
            relative_path: scan path relative to the database path
            mode: train, valid or test mode (dataset)

        Methods:
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
