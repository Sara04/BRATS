"""Toolchain for algorithm's training."""
import argparse
import sys
import os


from create_modules_objects import create_modules_objects_from_config


def main():
    """Function that runs training of a BRATS algorithm."""
    # _______________________________________________________________________ #
    parser = argparse.ArgumentParser(description='An algorithm for '
                                     'Brain Tumor Segmentation Challenge')

    parser.add_argument('-config', dest='config_path', required=True,
                        help='Path to the configuration file.')
    parser.add_argument('-o', dest='exp_out', required=True,
                        help='Path where the intermediate and final '
                             'results would be stored.')
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        print "\nConfiguration file does not exist!\n"
        sys.exit(2)

    if not os.path.exists(args.exp_out):
        try:
            os.mkdir(args.exp_out)
        except:
            print "\nOutput directory cannot be created!\n"

    modules_objects = create_modules_objects_from_config(args.config_path)
    # _______________________________________________________________________ #

    # 1. Loading modules' objects
    # _______________________________________________________________________ #
    db = modules_objects['database']
    prep = modules_objects['preprocessor']
    meta = modules_objects['meta_data_extractor']
    patch_ex = modules_objects['patch_extractor']
    seg = modules_objects['segmentator']
    post = modules_objects['postprocessor']
    # _______________________________________________________________________ #

    # 2. Loading training data and creating train valid split
    # _______________________________________________________________________ #
    db.load_training_dict()
    db.train_valid_split()
    # _______________________________________________________________________ #

    # 3. Computing preprocessing parameters
    # _______________________________________________________________________ #
    print "Getting preprocessing parameters..."
    prep.get_preprocessing_parameters(db, args.exp_out, 'train')
    # _______________________________________________________________________ #

    # 4. Computing brain masks and tumor distance maps
    # _______________________________________________________________________ #
    print "Computing brain masks and tumor distance maps..."
    meta.compute_brain_masks(db, args.exp_out, 'train')
    meta.compute_tumor_distance_maps(db, args.exp_out)
    # _______________________________________________________________________ #

    # 5. Normalize volumes
    # _______________________________________________________________________ #
    print "Volume normalization..."
    meta.compute_normalized_volumes(db, prep, args.exp_out, 'train')
    # _______________________________________________________________________ #

    # 6. Segmentator training and validation on training dataset
    # _______________________________________________________________________ #
    print "Segmentator training and validation..."
    seg.training_and_validation(db, prep, patch_ex)
    # _______________________________________________________________________ #

    # 7. Segmentator validation in terms of Dice scores on training subset
    # _______________________________________________________________________ #
    print "Segmentator Dice score validation..."
    seg.evaluate_dice_scores(db, prep, patch_ex)
    # _______________________________________________________________________ #

    # 8. Determine postprocessing parameters
    # _______________________________________________________________________ #
    post.determine_parameters(db)

if __name__ == '__main__':
    main()
