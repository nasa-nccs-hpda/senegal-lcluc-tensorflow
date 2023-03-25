import sys
import time
import logging
import argparse
from senegal_lcluc_tensorflow.model.pipelines.landuse_classification \
    import LandUseClassification as LandUsePipeline


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
                        required=False,
                        dest='config_file',
                        help='Path to the configuration file')

    parser.add_argument('-d',
                        '--label-regex',
                        type=str,
                        required=False,
                        default=None,
                        dest='label_regex',
                        help='Regex to label GeoTiff to download Sentinel for')

    parser.add_argument('-a',
                        '--gee-account',
                        type=str,
                        required=False,
                        default=None,
                        dest='gee_account',
                        help='GEE account to use to query data')

    parser.add_argument('-k',
                        '--gee-key',
                        type=str,
                        required=False,
                        default=None,
                        dest='gee_key',
                        help='GEE key to use with account to query data')

    parser.add_argument(
                        '-s',
                        '--step',
                        type=str,
                        nargs='*',
                        required=True,
                        dest='pipeline_step',
                        help='Pipeline step to perform',
                        default=['setup', 'preprocess', 'train', 'predict'],
                        choices=['setup', 'preprocess', 'train', 'predict'])

    args = parser.parse_args()

    # Setup timer to monitor script execution time
    timer = time.time()

    # Initialize pipeline object
    pipeline = LandUsePipeline(args.config_file)

    # Regression CHM pipeline steps
    if "setup" in args.pipeline_step:
        pipeline.setup(args.label_regex, args.gee_account, args.gee_key)
    #if "preprocess" in args.pipeline_step:
    #    pipeline.preprocess()
    #if "train" in args.pipeline_step:
    #    pipeline.train()
    #if "predict" in args.pipeline_step:
    #    pipeline.predict()

    #logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
