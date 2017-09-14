"""Class for BRATS 2017 segmentation."""
import os


class SegmentatorBRATS(object):
    """Class for BRATS 2017 segmentation."""

    """
        Methods:
            training_and_validation: training and validation of
                algorithm on training dataset
            compute_classification_scores: computation of
                classification scores
            save_model: saving trained model
            restore_model: restoreing trained model
    """
    def training_and_validation(self, db, prep, patch_ex, exp_out):
        """Run slgorithm's training and validation."""
        """
            Arguments:
                db: DatabaseBRATS object
                prep: PreprocessorBRATS object
                patch_ex: PatchExtractorBRATS object
                exp_out: path to the experiment output
        """
        raise NotImplementedError()

    def _validate(self, db, prep, patch_ex, exp_out, subset):
        raise NotImplementedError()

    def _train(self, db, prep, patch_ex, exp_out):
        raise NotImplementedError()

    def _compute_clf_scores_per_scan(self, db, prep, patch_ex,
                                     exp_out, scan):
        raise NotImplementedError()

    def compute_classification_scores(self, db, prep, patch_ex,
                                      exp_out, dataset):
        """Compute classification scores on selected dataset."""
        """
            Arguments:
                db: DatabaseBRATS object
                prep: PreprocessorBRATS object
                patch_ex: PatchExtractorBRATS object
                exp_out: path to the experiment output
                subset: selection of subset
                    (valid, test or train (validation part))
        """
        if dataset == 'train':
            data_dict = {}
            for ts in db.train_valid:
                data_dict[ts] = db.train_dict[ts]
        elif dataset == 'valid':
            data_dict = db.valid_dict
        elif dataset == 'test':
            data_dict = db.test_dict

        clf_path = os.path.join(exp_out, 'classification_probabilities',
                                db.name(), prep.name(), patch_ex.name(),
                                self.name(), dataset)
        if not os.path.exists(clf_path):
            os.makedirs(clf_path)
        for s in data_dict:
            self._compute_clf_scores_per_scan(db, prep, patch_ex, clf_path, s)

    def save_model(self, output_path, it=0):
        """Saving trained segmentation model."""
        """
            Arguments:
                output_path: path to the output directory
                it: iteration number if algorithm's training is iterative
        """
        raise NotImplementedError()

    def restore_model(self, input_path, it=0):
        """Restoring trained segmentation model."""
        """
            Arguments:
                input_path: path to the input directory
                it: iteration number if algorithm's training is iterative
        """
        raise NotImplementedError()
