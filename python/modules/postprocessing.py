"""Class for BRATS 2017 data postprocessing."""


class PostprocessorBRATS(object):
    """Class for BRATS 2017 data postprocessing."""

    def determine_parameters(self, db):
        raise NotImplementedError()

    def postprocess_valid(self, db):
        raise NotImplementedError()

    def postprocess_test(self, db):
        raise NotImplementedError()

    def name(self):
        """Class name reproduction."""
        return "%s()" % (type(self).__name__)
