# -*- coding: utf-8 -*-
"""
Logging for reVX
"""
import os
from reV.utilities.loggers import LoggingAttributes, setup_logger, FORMAT

REVX_LOGGERS = LoggingAttributes()


def init_logger(logger_name, log_level="INFO", log_file=None,
                log_format=FORMAT):
    """
    Starts logging instance and adds logging attributes to REV_LOGGERS

    Parameters
    ----------
    logger_name : str
        Name of logger to initialize
    log_level : str
        Level of logging to capture, must be key in LOG_LEVEL. If multiple
        handlers/log_files are requested in a single call of this function,
        the specified logging level will be applied to all requested handlers.
    log_file : str | list
        Path to file to use for logging, if None use a StreamHandler
        list of multiple handlers is permitted
    log_format : str
        Format for loggings, default is FORMAT

    Returns
    -------
    logger : logging.logger
        logging instance that was initialized
    """
    kwargs = {"log_level": log_level, "log_file": log_file,
              "log_format": log_format}
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
