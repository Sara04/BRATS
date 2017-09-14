"""Class for BRATS 2017 patch extraction."""


class PatchExtractorBRATS(object):
    """Class for BRATS 2017 patch extraction."""

    """
        Attributes:
            w, h, d: volume's widht, height and depth (number of slices)
            augment_train: flag indicating whether to augment training data

        Methods:
            extract_train_or_valid_data: extract data for
                training or validation (with equal distribution of classes)
            extract_test_patches: extract data for testing
    """
    def __init__(self, w=240, h=240, d=155, augment_train=False):
        """Initialization of PatchExtractorBRATS attributes."""
        self.w, self.h, self.d = [w, h, d]
        self.augment_train = augment_train in ['True', 'true', 'yes', 'Yes']

    def extract_train_or_valid_data(self, db, pp, seg, exp_out, mode='train'):
        """Extraction of training and validation data."""
        """
            Arguments:
                db: DatabaseBRATS object
                pp: PreprocessorBRATS object
                seg: SegmentatorBRATS object
                exp_put: path to the experiment output
                mode: train, valid or train_valid
                    valid and train_valid modes are without augmentation
            Returns:
                data and corresponding labels
        """
        raise NotImplementedError()

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
        raise NotImplementedError()
