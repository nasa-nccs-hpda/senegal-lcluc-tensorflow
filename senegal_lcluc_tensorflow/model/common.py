import sys
import tqdm
import logging
import argparse
import omegaconf
from datetime import datetime
from tensorflow_caney.model.config.cnn_config import Config


# -------------------------------------------------------------------------
# read_config
# -------------------------------------------------------------------------
def read_config(filename: str, config_class=Config):
    """
    Read configuration filename and initiate objects
    """
    # Configuration file initialization
    schema = omegaconf.OmegaConf.structured(config_class)
    conf = omegaconf.OmegaConf.load(filename)
    try:
        conf = omegaconf.OmegaConf.merge(schema, conf)
    except BaseException as err:
        sys.exit(f"ERROR: {err}")
    return conf


# -------------------------------------------------------------------------
# validate_date
# -------------------------------------------------------------------------
def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)


# -------------------------------------------------------------------------
# TqdmLoggingHandler
# -------------------------------------------------------------------------
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

# -------------------------------------------------------------------------
# get_tappan_base_command
# -------------------------------------------------------------------------
def get_tappan_base_command(dataset):

    return 'gdalwarp ' + \
        ' -multi' + \
        ' -tr 2 2' + \
        ' -s_srs "' + \
        dataset.GetSpatialRef().ExportToProj4() + \
        '"'
