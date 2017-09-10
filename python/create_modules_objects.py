"""Creating a dictionary of modules' objects from configuration file."""
import sys
import json

from modules.database import DatabaseBRATS
from modules.preprocessing import PreprocessorBRATS
from modules.patch_extraction import PatchExtractorBRATS
from modules.segmentation import SegmentatorBRATS
from modules.postprocessing import PostprocessorBRATS

import modules.preprocessors
import modules.patch_extractors
import modules.segmentators
import modules.postprocessors

required_modules =\
    {
        'database':
        {
            'mother_class': DatabaseBRATS,
            'child_classes': None,
        },
        'preprocessor':
        {
            'mother_class': PreprocessorBRATS,
            'child_classes': modules.preprocessors,
        },
        'patch_extractor':
        {
            'mother_class': PatchExtractorBRATS,
            'child_classes': modules.patch_extractors
        },
        'segmentator':
        {
            'mother_class': SegmentatorBRATS,
            'child_classes': modules.segmentators
        },
        'postprocessor':
        {
            'mother_class': PostprocessorBRATS,
            'child_classes': modules.postprocessors
        }
    }
optional_modules = {}


def _config_file_error():
    print 'Invalid configuration file.'
    print 'Configuration file must contain dictionary of modules '
    print 'with class names and corresponding arguments.'
    sys.exit(2)


def _missing_module_error(m):
    print 'Module <<%s>> is missing and it is required.' % m
    sys.exit(2)


def _invalid_module_name(m):
    print 'Module name <<%s>> is invalid.' % m
    sys.exit(2)


def _non_existing_child_class(child_class, mother_class):
    print ("Class <<%s>> is not child class of <<%s>> "
           % (child_class.__name__, mother_class.__name__))
    sys.exit(2)


def create_modules_objects_from_config(config_path):
    """Creating a dictionary of modules' objects."""
    """
        Arguments:
            config_path: path to the configuration file
        Returns:
            dictionary of modules objects
    """
    try:
        with open(config_path, 'r') as f_config:
            configuration = json.load(f_config)
    except:
        _config_file_error()

    if 'modules' not in configuration:
        _config_file_error()

    for rm in required_modules:
        if rm not in configuration['modules']:
            _missing_module_error(rm)

    all_modules = required_modules
    required_modules.update(optional_modules)

    objects = {}
    for m in configuration['modules']:
        if m not in all_modules:
            _invalid_module_name(m)

        mother_class = all_modules[m]['mother_class']
        child_classes = all_modules[m]['child_classes']

        if child_classes:
            module_class = getattr(child_classes,
                                   configuration['modules'][m]['name'])
            if not issubclass(module_class, mother_class):
                _non_existing_child_class(module_class, mother_class)
        else:
            module_class = mother_class

        module_obj = module_class(**configuration['modules'][m]['arguments'])
        objects[m] = module_obj

    return objects
