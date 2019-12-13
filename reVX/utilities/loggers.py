# -*- coding: utf-8 -*-
"""
Logging for reVX
"""
import os
from reV.utilities.loggers import LoggingAttributes, setup_logger

REVX_LOGGERS = LoggingAttributes()


def init_logger(logger_name, **kwargs):
    """
    Starts logging instance and adds logging attributes to REV_LOGGERS

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


def init_mult(name, logdir, modules, verbose=False, node=False):
    """Init multiple loggers to a single file or stdout.

    Parameters
    ----------
    name : str
        Job name; name of log file.
    logdir : str
        Target directory to save .log files.
    modules : list | tuple
        List of reV modules to initialize loggers for.
    verbose : bool
        Option to turn on debug logging.
    node : bool
        Flag for whether this is a node-level logger. If this is a node logger,
        and the log level is info, the log_file will be None (sent to stdout).

    Returns
    -------
    loggers : list
        List of logging instances that were initialized.
    """

    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    loggers = []
    for module in modules:
        log_file = os.path.join(logdir, '{}.log'.format(name))

        # check for redundant loggers in the REV_LOGGERS singleton
        logger = REVX_LOGGERS[module]

        if ((not node or (node and log_level == 'DEBUG'))
                and 'log_file' not in logger):
            # No log file belongs to this logger, init a logger file
            logger = init_logger(module, log_level=log_level,
                                 log_file=log_file)
        elif node and log_level == 'INFO':
            # Node level info loggers only go to STDOUT/STDERR files
            logger = init_logger(module, log_level=log_level, log_file=None)
        loggers.append(logger)

    return loggers
