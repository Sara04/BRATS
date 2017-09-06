"""Class for BRATS 2017 data preprocessing."""

import numpy as np
import json
import os


class PreprocessorBRATS(object):
    """Class for BRATS 2017 data preprocessing."""

    """
        Attributes:
            norm_type: normalization type (mean_std or min_max)

            train_norm_params: dictionary for train normalization parameters
            valid_norm_params: dictionary for valid normalization parameters
            test_norm_params: dictionary for test normalization parameters

            clip: flag indicating whether to clip values after normalization
            clip_u: upper clip value
            clip_l: lower clip value

        Methods:
            compute_norm_params: compute normalization parameters for
                given volume
            get_preprocessing_parameters: get preprocessing parameters for
                selected database (train, valid or test)
            normalize: normalize data within given mask
            normalize_volumes: normalize all volumes of a scan within
                brain and skul mask, and clip if required
            name: reproduce PreprocessorBRATS object's name
    """
    def __init__(self, norm_type='mean_std',
                 clip=True, clip_l=-2.0, clip_u=2.0):
        """Initialization of PreprocessorBRATS attributes."""
        self.norm_type = norm_type
        self.train_norm_params = {}
        self.valid_norm_params = {}
        self.test_norm_params = {}
        self.clip = clip
        self.clip_l = clip_l
        self.clip_u = clip_u

    def _compute_mean_std_params(self, volume):
        non_zeros = volume != 0.0
        mean_ = np.mean(volume[non_zeros == 1])
        std_ = np.std(volume[non_zeros == 1])

        return {'mean': float(mean_), 'std': float(std_)}

    def _compute_min_max_params(self, volume):
        max_ = np.max(volume[volume != 0])
        min_ = np.min(volume[volume != 0])

        return {'min': float(min_), 'max': float(max_)}

    def compute_norm_params(self, volume):
        """Normalization parameters computation."""
        """
            Arguments:
                volume: input volume

            Returns:
                dictionary of computed parameters
        """
        if self.norm_type == 'mean_std':
            return self._compute_mean_std_params(volume)
        if self.norm_type == 'min_max':
            return self._compute_min_max_params(volume)

    def _compute_preprocess_parameters(self, db, data_dict):
        n_params = {}
        for s in data_dict:
            for m in data_dict[s].modalities:
                if m in['seg']:
                    continue

                volume = data_dict[s].load_volume(db, m)
                volume_norm_params = self.compute_norm_params(volume)

                if s not in n_params:
                    n_params[s] = {}
                if m not in n_params[s]:
                    n_params[s][m] = {}

                for p in volume_norm_params:
                    n_params[s][m][p] = volume_norm_params[p]
        return n_params

    def _load_preprocess_parameters(self, params_output_path):
        with open(params_output_path, 'r') as f:
            return json.load(f)

    def _save_preprocess_parameters(self, params_output_dir, data_dict):
        pp_done_path = os.path.join(params_output_dir, 'done')
        pp_params_output_path = os.path.join(params_output_dir, 'params.json')

        if not os.path.exists(params_output_dir):
            os.makedirs(params_output_dir)
        with open(pp_params_output_path, 'w') as f:
            json.dump(data_dict, f)
        with open(pp_done_path, 'w') as f:
            f.close()

    def get_preprocessing_parameters(self, db, exp_out, mode):
        """Getting a dictionary of preprocessing parameters."""
        """
            Arguments:
                db: DatabaseBRATS
                exp_out: path to the experiment meta output
                mode: training or validation subsets
        """
        if mode == 'train':
            data_dict = db.train_dict
        elif mode == 'valid':
            data_dict = db.valid_dict
        elif mode == 'test':
            data_dict = db.test_dict

        out_path = os.path.join(exp_out, 'preprocessing', self.name(), mode)
        done_path = os.path.join(out_path, 'done')

        if os.path.exists(done_path):
            params_out_path = os.path.join(out_path, 'params.json')
            n_params = self._load_preprocess_parameters(params_out_path)
        else:
            n_params = self._compute_preprocess_parameters(db, data_dict)
            self._save_preprocess_parameters(out_path, n_params)

        if mode == 'train':
            self.train_norm_params = n_params
        elif mode == 'valid':
            self.valid_norm_params = n_params
        elif mode == 'test':
            self.test_norm_params = n_params

    def normalize(self, scan, m, v, mask):
        """Normalization of the input volume array v."""
        """
            Arguments:
                scan: ScanBRATS object
                m: modality
                v: data/volume
                mask: brain and skul mask
            Returns:
                normalized data
        """
        if scan.name in self.train_norm_params:
            n_params = self.train_norm_params[scan.name][m]
        elif scan.name in self.valid_norm_params:
            n_params = self.valid_norm_params[scan.name][m]
        elif scan.name in self.test_norm_params:
            n_params = self.test_norm_params[scan.name][m]

        if self.norm_type == 'mean_std':
            return mask * (v - n_params['mean']) / n_params['std']
        if self.norm_type == 'min_max':
            return (mask * (v - n_params['min']) /
                    (n_params['max'] - n_params['max']))

    def normalize_volumes(self, db, scan):
        """Normalization of all volumes."""
        """
            Arguments:
                db: DatabaseBRATS object
                scan: ScanBRATS object
            Returns:
                list of normalized volumes
        """
        volumes = scan.load_volumes(db)
        volumes_n = {}
        for m_idx, m in enumerate(db.modalities[:-1]):
            volumes_n[m] = self.normalize(scan, m, volumes[m_idx], volumes[5])
            if self.clip:
                volumes_n[m] = np.clip(volumes_n[m], self.clip_l, self.clip_u)
        return volumes_n

    def name(self):
        """Class name reproduction."""
        """
            Returns:
                PreprocessingBRATS object's name
        """
        return "%s(norm_type=%s)" % (type(self).__name__, self.norm_type)
