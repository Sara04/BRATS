"""Class for BRATS 2017 data preprocessing."""


class PreprocessorBRATS(object):
    """Class for BRATS 2017 data preprocessing."""

    """
        Methods:
            get_preprocessing_parameters: creates a dictionary
                of parameters for volume preprocessing
            name: reproduce PreprocessorBRATS object's name
    """
    def get_preprocessing_parameters(self):
        """Get dictionary of preprocessing parameters per scan."""
        raise NotImplementedError()

    def name(self):
        """Class name."""
        raise NotImplementedError()
