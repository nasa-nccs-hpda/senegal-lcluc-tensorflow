import sys
import time
import logging
import argparse
from senegal_lcluc_tensorflow.model.pipelines.tappan_pipeline \
    import main as tappan_main


# -----------------------------------------------------------------------------
# main
#
# python tappan_pipeline_cli.py -c config.yaml
# -----------------------------------------------------------------------------
def main():

    # Process command-line args.
    desc = 'Use this application to generate Tappan squares and Clusters.'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        dest='config_file',
                        help='Path to the configuration file (YAML)')

    args = parser.parse_args()

    # Setup timer to monitor script execution time
    timer = time.time()

    # Call main function with parsed arguments
    tappan_main(args.config_file)

    logging.info(f'Took {(time.time()-timer)/60.0:.2f} min.')


# -----------------------------------------------------------------------------
# Invoke the main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
