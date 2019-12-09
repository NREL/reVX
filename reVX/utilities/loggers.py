# -*- coding: utf-8 -*-
"""
Logging for reV_X
"""
# pylint: disable-msg=W0611
from reV.utilities.loggers import LoggingAttributes, setup_logger, init_mult

REVX_LOGGERS = LoggingAttributes()


def init_logger(logger_name, **kwargs):
    """
    Starts logging instance and adds logging attributes to REVX_LOGGERS

    Parameters
    ----------
    logger_name : str
        Name of logger to initialize
    **kwargs
        Logging attributes used to setup_logger

    Returns
    -------
    logger : logging.logger
        logging instance that was initialized
    """
    logger = setup_logger(logger_name, **kwargs)

    REVX_LOGGERS[logger_name] = kwargs

    return logger
