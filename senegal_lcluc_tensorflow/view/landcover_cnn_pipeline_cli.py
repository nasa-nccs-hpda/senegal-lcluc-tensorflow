import sys
import time
import logging
import argparse
from senegal_lcluc_tensorflow.model.pipelines.landcover_pipeline \
    import LandCoverPipeline


# -----------------------------------------------------------------------------
# main
#
# python landcover_cnn_pipeline_cli.py -c config.yaml -d config.yaml -s train
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to perform CNN segmentation.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-d',
                        '--data-csv',
                        type=str,
                        required=False,
                        dest='data_csv',
                        help='Path to the data configuration file')

    parser.add_argument('-t',
                        '--test-truth-regex',
                        type=str,
                        required=False,
                        dest='test_truth_regex',
                        help='Path to test truth regex')

    parser.add_argument('-v',
                        '--validation-database',
                        type=str,
                        required=False,
                        default=None,
                        dest='validation_database',
                        help='Path to validation database')

    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=['preprocess', 'train', 'predict'],
                        choices=[
                            'preprocess', 'train', 'predict',
                            'test', 'validate'
                        ])

    args = parser.parse_args()

    # Setup timer to monitor script execution time
    timer = time.time()

    # Initialize pipeline object
    pipeline = LandCoverPipeline(args.config_file, args.data_csv)

    # Regression CHM pipeline steps
    if "preprocess" in args.pipeline_step:
        pipeline.preprocess(enable_multiprocessing=True)
    if "train" in args.pipeline_step:
        pipeline.train()
    if "predict" in args.pipeline_step:
        pipeline.predict()
    if "test" in args.pipeline_step:
        pipeline.test(args.test_truth_regex)
    if "validate" in args.pipeline_step:
        pipeline.validate(args.validation_database)

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
