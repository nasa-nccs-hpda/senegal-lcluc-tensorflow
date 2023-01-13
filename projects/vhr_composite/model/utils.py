import datetime
import logging
import tqdm


def get_post_str():
    sdtdate = datetime.datetime.now()
    year = sdtdate.year
    hm = sdtdate.strftime('%H%M')
    sdtdate = sdtdate.timetuple()
    jdate = sdtdate.tm_yday
    post_str = '{}{:03}{}'.format(year, jdate, hm)
    return post_str


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
