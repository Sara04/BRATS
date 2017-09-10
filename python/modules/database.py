"""Class for BRATS 2017 database management."""
import os
import natsort as ns
import numpy as np

from scan import ScanBRATS


class DatabaseBRATS(object):
    """Class for BRATS 2017 database management."""

    """
        Attributes:
            db_path: path where training and validation subsets are stored
            n_classes: number of tumor subregions
            classes: list of tumor subregions, namely
                0 - non tumorous tissue
                1 -  necrotic and non-enhancing tumor
                2 - peritumoral edema
                4 - enhancing tumor
            n_modalities: number of MRI modalities
            modalities: list of MRI modalities, namely
                t1 - T1 weighted image, spin-lattice relaxation time
                t2 - T2 weighted image, spin-splin relaxation time
                t1ce - contrast enhanced T1 weighted image
                flair - T2 weighted fluid attenuation inversion recovery

            h, w, d: scan's height, width and depth (number of slices)

            valid_p: percentage of training data that will be used for
                algorithm training validation

            train_dict: dictionary for storing training scans
            valid_dict: dictionary for storing validation scans
            test_dict: dictionary for storing test scans

            train_train: list for storing scans used for algorithm training
            train_valid: list for storing scans used for algorithm validation

            Directories:
            brain_masks_dir: path to the brain masks
            normalized_volumes_dir: path to the normalized volumes
            tumor_dist_dir: path to the tumor distance maps
            seg_scores_dir: path to the segmentation scores
            seg_results_dir: path to the segmentation results

        Methods:
            load_training_dict: creating a dictionary of training scans
            load_validation_dict: creating a dictionary of validation scans
            load_test_dict: creating a dictionary of testing scans
            train_valid_split: split training database into train and valid
                subsets (validation dataset is used for evaluation)
            name: returns database name with train valid split parameter
    """
    def __init__(self, db_path, n_classes=4, classes=[0, 1, 2, 4],
                 n_modalities=4, modalities=['t1', 't2', 't1ce', 'flair'],
                 h=240, w=240, d=155, valid_p=0.2):
        """Initialization of DatabaseBRATS attributes."""
        self.db_path = db_path
        self.n_classes = n_classes
        self.classes = classes
        self.n_modalities = n_modalities
        self.modalities = modalities

        self.h, self.w, self.d = [h, w, d]

        self.valid_p = valid_p

        self.train_dict = {}
        self.valid_dict = {}
        self.test_dict = {}

        self.train_train = []
        self.train_valid = []

        self.brain_masks_dir = None
        self.normalized_volumes_dir = None
        self.tumor_dist_dir = None
        self.seg_results_dir = None
        self.seg_results_dir = None

    def load_training_dict(self, folder_name='Brats17TrainingData'):
        """Creating a dictionary of training HGG and LGG scans."""
        """
            Arguments:
                folder_name: folder where the training data is stored
        """
        grades = ['HGG', 'LGG']

        for g in grades:
            scans = os.listdir(os.path.join(self.db_path, folder_name, g))
            scans = ns.natsort(scans)
            for s in scans:
                if s not in self.train_dict:
                    s_relative_path = os.path.join(folder_name, g, s)
                    self.train_dict[s] = ScanBRATS(s, s_relative_path, 'train')

    def load_validation_dict(self, folder_name='Brats17ValidationData'):
        """Creating a dictionary of validation scans."""
        """
            Arguments:
                folder_name: folder where the validation data is stored
        """
        scans = os.listdir(os.path.join(self.db_path, folder_name))
        for s in scans:
            if s.endswith('csv'):
                continue
            if s not in self.valid_dict:
                s_relative_path = os.path.join(folder_name, s)
                self.valid_dict[s] = ScanBRATS(s, s_relative_path, 'valid')

    def load_test_dict(self, folder_name='Brats17TestingData'):
        """Creating a dictionary of testing scans."""
        """
            Arguments:
                folder_name: folder where the testing data is stored
        """
        scans = os.listdir(os.path.join(self.db_path, folder_name))
        for s in scans:
            if s.endswith('csv'):
                continue
            if s not in self.test_dict:
                s_relative_path = os.path.join(folder_name, s)
                self.test_dict[s] = ScanBRATS(s, s_relative_path, 'test')

    def train_valid_split(self, folder_name='Brats17TrainingData'):
        """Splitting training data into train and valid subsets."""
        """
            Note:
                Training data is split into train and validation subset since
                validation database is used for final testing using BRATS
                evaluation system and ground truth labels are not provided
                for it.
        """
        """
            Arguments:
                folder_name: name of the folder where training data is stored
        """
        np.random.seed(123456)

        lgg_scans = os.listdir(os.path.join(self.db_path, folder_name, 'LGG'))
        n_lgg_scans = len(lgg_scans)
        lgg_select_valid = np.random.choice(n_lgg_scans,
                                            int(np.round(n_lgg_scans *
                                                         self.valid_p)),
                                            replace=False)

        hgg_scans = os.listdir(os.path.join(self.db_path, folder_name, 'HGG'))
        n_hgg_scans = len(hgg_scans)
        hgg_select_valid = np.random.choice(n_hgg_scans,
                                            int(np.round(n_hgg_scans *
                                                         self.valid_p)),
                                            replace=False)

        for l_idx, l in enumerate(lgg_scans):
            if l_idx in lgg_select_valid:
                self.train_valid.append(l)
            else:
                self.train_train.append(l)

        for h_idx, h in enumerate(hgg_scans):
            if h_idx in hgg_select_valid:
                self.train_valid.append(h)
            else:
                self.train_train.append(h)

    def name(self):
        """Return database name."""
        return "%s(valid_p=%s)" % (type(self).__name__, self.valid_p)
