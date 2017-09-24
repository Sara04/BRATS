"""Class for BRATS 2017 patch extraction."""

import numpy as np
from .. import PatchExtractorBRATS


class PatchExtractorBRATSForCNN12(PatchExtractorBRATS):
    """Class for BRATS 2017 patch extraction for cnn segmentators."""

    """
        Attributes:
            w, h, d: volume's widht, height and depth (number of slices)
            lp_w, lp_h, lp_d: width, height and depth of large region
                patches
            sp_w, sp_h, sp_d: width, height and depth of small region
                patches
            td_th_1, td_th_2: thresholds used to determine small and large
                neighbourhoods of tumor
            lpm_d: depth of large region patches per modality
            spm_d: depth of small region patches per modality
            augment_train: flag indicating whether to augment training data

        Methods:
            extract_train_or_valid_data: extract data for train or valid
                (with equal distribution of classes)
            extract_test_patches: extract data for testing
    """
    def __init__(self,
                 scans_per_batch=5, patches_per_scan=50,
                 lp_w=45, lp_h=45, lp_d=11, sp_w=17, sp_h=17, sp_d=4,
                 td_th_1=20, td_th_2=256, lpm_d=2, spm_d=1, **kwargs):
        """Initialization of PatchExtractorBRATS attributes."""
        super(PatchExtractorBRATSForCNN12, self).__init__(**kwargs)
        self.scans_per_batch = scans_per_batch
        self.patches_per_scan = patches_per_scan

        self.lp_w, self.lp_h, self.lp_d = [lp_w, lp_h, lp_d]
        self.sp_w, self.sp_h, self.sp_d = [sp_w, sp_h, sp_d]

        self.td_th_1 = td_th_1
        self.td_th_2 = td_th_2

        self.pvs, self.pve = [(self.lp_h - 1) / 2, (self.lp_h + 1) / 2]
        self.phs, self.phe = [(self.lp_w - 1) / 2, (self.lp_w + 1) / 2]

        self._get_coordinates()

        self.lpm_d = lpm_d
        self.spm_d = spm_d

    def _get_coordinates(self):
        self.h_coord = np.zeros((self.h, self.w))
        self.v_coord = np.zeros((self.h, self.w))
        self.d_coord = np.zeros((self.h, self.w, self.d))
        wh, hh, dh = [self.w / 2, self.h / 2, self.d / 2]
        for r_idx in range(self.h):
            for c_idx in range(self.w):
                self.h_coord[r_idx, c_idx] = float(c_idx - wh) / wh
                self.v_coord[r_idx, c_idx] = float(r_idx - hh) / hh
                for d_idx in range(self.d):
                    self.d_coord[r_idx, c_idx, d_idx] = float(d_idx - dh) / dh

    def _allocate_data_memory(self, db):
        data = {'region_1': {}, 'region_2': {}}
        for i in db.classes:
            data['region_1'][i] = {}
            data['region_1'][i]['l_patch'] =\
                np.zeros((self.patches_per_scan * self.scans_per_batch,
                          self.lp_w * self.lp_h * self.lp_d))
            data['region_1'][i]['s_patch'] =\
                np.zeros((self.patches_per_scan * self.scans_per_batch,
                          self.sp_w * self.sp_h * self.sp_d))
        for i in range(2):
            data['region_2'][i] = {}
            data['region_2'][i]['l_patch'] =\
                np.zeros((self.patches_per_scan * self.scans_per_batch,
                          self.lp_w * self.lp_h * self.lp_d))
            data['region_2'][i]['s_patch'] =\
                np.zeros((self.patches_per_scan * self.scans_per_batch,
                          self.sp_w * self.sp_h * self.sp_d))
        return data

    def _extract_distances_for_point(self, b):
        dist = np.zeros((self.lp_h, self.lp_w, 3))

        b_, d_ = [np.copy(b), [0, self.lp_h, 0, self.lp_w]]
        b_, d_ = self._verify_border_cases(b_, d_)

        dist[d_[0]:d_[1], d_[2]:d_[3], 0] =\
            self.h_coord[b[0]: b[1], b[2]: b[3]]
        dist[d_[0]:d_[1], d_[2]:d_[3], 0] =\
            self.v_coord[b[0]: b[1], b[2]: b[3]]
        dist[d_[0]:d_[1], d_[2]:d_[3], 0] =\
            self.d_coord[b[0]: b[1], b[2]: b[3], b[4]]
        return dist

    def _select_scans_randomly(self, db, mode):
        if mode.startswith('train'):
            scans = db.train_train
        elif mode == 'valid':
            scans = db.train_valid
        rs = np.random.choice(len(scans), self.scans_per_batch, replace=False)
        return [scans[idx] for idx in rs]

    def _verify_border_cases(self, b_, d_):

        if b_[0] < 0:
            d_[0], b_[0] = [0 - b_[0], 0]
        if b_[2] < 0:
            d_[2], b_[2] = [0 - b_[2], 0]
        if b_[1] > self.h:
            d_[1], b_[1] = [self.lp_h - (b_[1] - self.h), self.h]
        if b_[3] > self.w:
            d_[3], b_[3] = [self.lp_w - (b_[3] - self.w), self.w]

        return b_, d_

    def _modality_patches(self, scan, m, volume, b, mode):

        lpm = np.zeros((self.lp_h, self.lp_w, self.lpm_d))
        spm = np.zeros((self.sp_h, self.sp_w, self.spm_d))

        b_, d_ = [np.copy(b), [0, self.lp_h, 0, self.lp_w]]
        b_, d_ = self._verify_border_cases(b_, d_)

        lpm[d_[0]:d_[1], d_[2]:d_[3], 0] =\
            volume[b_[0]: b_[1], b_[2]: b_[3], b_[4]]

        if mode == 'train' and self.augment_train:
            # Shift mirrored patches for augmentation
            b_, d_ = [np.copy(b), [0, self.lp_h, 0, self.lp_w]]
            sh_1, sh_2 = [np.random.randint(11) - 5, np.random.randint(11) - 5]
            b_ = [b_[0] + sh_1, b_[1] + sh_1, b_[2] + sh_2, b_[3] + sh_2, b_[4]]
            b_, d_ = self._verify_border_cases(b_, d_)

        lpm[self.lp_h - d_[1]:self.lp_h - d_[0],
            self.lp_w - d_[3]:self.lp_w - d_[2], 1] =\
            volume[self.h - b_[1]: self.h - b_[0], b_[2]: b_[3], b_[4]]

        spm[:, :, 0] = lpm[:, :, 0][(self.lp_h - 1) / 2 - (self.sp_h - 1) / 2:
                                    (self.lp_h - 1) / 2 + (self.sp_h + 1) / 2,
                                    (self.lp_w - 1) / 2 - (self.sp_w - 1) / 2:
                                    (self.lp_w - 1) / 2 + (self.sp_w + 1) / 2]
        return lpm, spm

    def _class_patches(self, db, scan, volumes, mask, pp, mode):
        lpc = np.zeros((self.patches_per_scan,
                        self.lp_h * self.lp_w * self.lp_d))
        spc = np.zeros((self.patches_per_scan,
                        self.sp_h * self.sp_w * self.sp_d))

        n_available = len(mask[0])
        if n_available:
            n_select = np.min([self.patches_per_scan, n_available])
            select = np.random.choice(n_available, n_select, replace=False)
            for s_idx, s in enumerate(select):
                lp = np.zeros((self.lp_h, self.lp_w, self.lp_d))
                sp = np.zeros((self.sp_h, self.sp_w, self.sp_d))
                bb = [mask[0][s] - self.pvs, mask[0][s] + self.pve,
                      mask[1][s] - self.phs, mask[1][s] + self.phe,
                      mask[2][s]]
                for i, m in enumerate(db.modalities):
                    lpm, spm =\
                        self._modality_patches(scan, m, volumes[i], bb, mode)
                    lp[:, :, i * self.lpm_d:(i + 1) * self.lpm_d] = lpm
                    sp[:, :, i * self.spm_d:(i + 1) * self.spm_d] = spm
                lp[:, :, self.lpm_d * db.n_modalities:
                   (self.lpm_d * db.n_modalities + 3)] =\
                    self._extract_distances_for_point(bb)

                if mode == 'train' and self.augment_train:
                    flip = np.random.randint(2)
                    if flip:
                        lp[:, :, :-3] = lp[::-1, :, :-3]
                        sp[:, :, :] = sp[::-1, :, :]
                        lp[:, :, -2] = lp[::-1, :, -2] * (-1)

                    for lp_idx in range(self.lp_d - 3):
                        lp[:, :, lp_idx] += 0.2 * np.random.randn(1)[0]
                    for sp_idx in range(self.sp_d):
                        sp[:, :, sp_idx] += 0.2 * np.random.randn(1)[0]
                lpc[s_idx, :] = np.ravel(lp)
                spc[s_idx, :] = np.ravel(sp)
            n_available = s_idx + 1
        return lpc[0:n_available, :], spc[0:n_available]

    def _scan_patches(self, scan, db, prep, mode):
        ps = {'region_1': {}, 'region_2': {}}
        volumes = scan.load_volumes(db, load_normalized=True)

        tdm_1 = volumes[5] * (volumes[6] <= self.td_th_1)
        tdm_2 = volumes[5] * (volumes[6] <= self.td_th_2)

        for c in db.classes:
            mask_1 = np.where((volumes[4] == c) * tdm_1)
            ps['region_1'][c] = {}
            ps['region_1'][c]['l_patch'], ps['region_1'][c]['s_patch'] =\
                self._class_patches(db, scan, volumes, mask_1, prep, mode)
        for c in range(2):
            if c == 0:
                mask_2 = np.where((volumes[4] == 0) * tdm_2)
            else:
                mask_2 = np.where((volumes[4] != 0) * tdm_2)
            ps['region_2'][c] = {}
            ps['region_2'][c]['l_patch'], ps['region_2'][c]['s_patch'] =\
                self._class_patches(db, scan, volumes, mask_2, prep, mode)
        return ps

    def _shuffle_and_select_data(self, data_dict, c, db):
        c_min = min([c['r1'][k] for k in c['r1']])
        lp_data_r1 = np.zeros((db.n_classes * c_min,
                               self.lp_w * self.lp_h * self.lp_d))
        sp_data_r1 = np.zeros((db.n_classes * c_min,
                               self.sp_w * self.sp_h * self.sp_d))
        labels_r1 = np.zeros((db.n_classes * c_min, db.n_classes))

        for i, k in enumerate(c['r1'].keys()):
            p = np.arange(c['r1'][k])
            np.random.shuffle(p)
            lp_data_r1[i * c_min:(i + 1) * c_min, :] =\
                data_dict['region_1'][k]['l_patch'][p, :][0:c_min, :]
            sp_data_r1[i * c_min:(i + 1) * c_min, :] =\
                data_dict['region_1'][k]['s_patch'][p, :][0:c_min, :]
            labels_r1[i * c_min:(i + 1) * c_min, i] = 1

        c_min = min([c['r2'][k] for k in [0, 1]])
        lp_data_r2 = np.zeros((2 * c_min, self.lp_w * self.lp_h * self.lp_d))
        sp_data_r2 = np.zeros((2 * c_min, self.sp_w * self.sp_h * self.sp_d))
        labels_r2 = np.zeros((2 * c_min, 2))
        for i, k in enumerate(c['r2'].keys()):
            p = np.arange(c['r2'][k])
            np.random.shuffle(p)
            lp_data_r2[i * c_min:(i + 1) * c_min, :] =\
                data_dict['region_2'][k]['l_patch'][p, :][0:c_min, :]
            sp_data_r2[i * c_min:(i + 1) * c_min, :] =\
                data_dict['region_2'][k]['s_patch'][p, :][0:c_min, :]
            labels_r2[i * c_min:(i + 1) * c_min, i] = 1

        data_r1, data_r2 = [{}, {}]
        p1, p2 =\
            [np.arange(lp_data_r1.shape[0]), np.arange(lp_data_r2.shape[0])]
        np.random.shuffle(p1), np.random.shuffle(p2)

        data_r1['l_patch'], data_r1['s_patch'], data_r1['labels'] =\
            [lp_data_r1[p1, :], sp_data_r1[p1, :], labels_r1[p1, :]]

        data_r2['l_patch'], data_r2['s_patch'], data_r2['labels'] =\
            [lp_data_r2[p2, :], sp_data_r2[p2, :], labels_r2[p2, :]]

        return data_r1, data_r2

    def extract_train_or_valid_data(self, db, prep, exp_out, mode='train'):
        """Extraction of training data with augmentation."""
        """
            Arguments:
                db: DatabaseBRATS object
                pp: PreprocessorBRATS object
                exp_put: path to the experiment output
                mode: train, valid or train_test(without augmentation) mode
            Returns:
                data and corresponding labels
        """
        data = self._allocate_data_memory(db)
        selected_scans = self._select_scans_randomly(db, mode)

        data_dict = db.train_dict
        c = {'r1': {}, 'r2': {}}
        for k in db.classes:
            c['r1'][k] = 0
        for k in range(2):
            c['r2'][k] = 0

        for s_idx, s in enumerate(selected_scans):
            data_s = self._scan_patches(data_dict[s], db, prep, mode)
            for i in db.classes:
                n = data_s['region_1'][i]['l_patch'].shape[0]
                data['region_1'][i]['l_patch'][c['r1'][i]:c['r1'][i] + n, :] =\
                    data_s['region_1'][i]['l_patch']
                data['region_1'][i]['s_patch'][c['r1'][i]:c['r1'][i] + n, :] =\
                    data_s['region_1'][i]['s_patch']
                c['r1'][i] += n
            for i in range(2):
                n = data_s['region_2'][i]['l_patch'].shape[0]
                data['region_2'][i]['l_patch'][c['r2'][i]:c['r2'][i] + n, :] =\
                    data_s['region_2'][i]['l_patch']
                data['region_2'][i]['s_patch'][c['r2'][i]:c['r2'][i] + n, :] =\
                    data_s['region_2'][i]['s_patch']
                c['r2'][i] += n

        data_ = {}
        data_['region_1'], data_['region_2'] =\
            self._shuffle_and_select_data(data, c, db)

        if data_['region_1']['labels'].shape[0]:
            return data_
        else:
            return None

    def extract_test_patches(self, scan, db, pp, volumes, ind_part):
        """Extraction of test patches."""
        """
            Arguments:
                scan: selected scan
                db: DatabaseBRATS object
                pp: PreprocessorBRATS object
                volumes: scan volumes
                ind_part: list of voxel indices at which patches will be
                    extracted
            Returns:
                extracted test patches
        """
        n_indices = len(ind_part[0])
        test_data = {}
        test_data['l_patch'] =\
            np.zeros((n_indices, self.lp_h * self.lp_w * self.lp_d))
        test_data['s_patch'] =\
            np.zeros((n_indices, self.sp_h * self.sp_w * self.sp_d))

        lp = np.zeros((self.lp_h, self.lp_w, self.lp_d))
        sp = np.zeros((self.sp_h, self.sp_w, self.sp_d))
        for j in range(n_indices):
            b = [ind_part[0][j] - self.pvs, ind_part[0][j] + self.pve,
                 ind_part[1][j] - self.phs, ind_part[1][j] + self.phe,
                 ind_part[2][j]]
            for i, m in enumerate(db.modalities):
                lpm, spm = self._modality_patches(scan, m, volumes[i], b)
                lp[:, :, i * self.lpm_d:(i + 1) * self.lpm_d] = lpm
                sp[:, :, i * self.spm_d:(i + 1) * self.spm_d] = spm
            lp[:, :, self.lpm_d * db.n_modalities:
               (self.lpm_d * db.n_modalities + 3)] =\
                self._extract_distances_for_point(b)
            test_data['l_patch'][j, :] = np.ravel(lp)
            test_data['s_patch'][j, :] = np.ravel(sp)
        return test_data

    def name(self):
        """Class name reproduction."""
        """
            Returns patch_extractor's name.
        """
        return ("%s(lp_w=%s, lp_h=%s, lp_d=%s, sp_w=%s, sp_h=%s, sp_d=%s, "
                "td_th_1=%s, td_th_2=%s)"
                % (type(self).__name__,
                   self.lp_w, self.lp_h, self.lp_d,
                   self.sp_w, self.sp_h, self.sp_d,
                   self.td_th_1, self.td_th_2))
