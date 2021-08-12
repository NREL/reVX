"""
Load default configuration for tie-line cost determination

Mike Bannister
7/27/21
"""
import logging
import os

from rex.utilities.utilities import safe_json_load

CONFIG = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                      'config.json')
logger = logging.getLogger(__name__)


class XmissionConfig(dict):
    """ Load default tie lines configuration """

    def __init__(self, config=None):
        """
        Use default values for all parameters if None.

        Parameters
        ----------
        config : str | dict, optional
            Path to json file containing Xmission cost configuration values,
            or jsonified dict of cost configuration values,
            or dictionary of configuration values,
            or dictionary of paths to config jsons,
            if None use defaults in './config.json'
        """
        super().__init__()

        self.update(safe_json_load(CONFIG))

        if config is not None:
            if isinstance(config, str):
                config = safe_json_load(config)

            if not isinstance(config, dict):
                msg = ('Xmission costs config must be a path to a json file, '
                       'a jsonified dictionary, or a dictionary, not: {}'
                       .format(config))
                logger.error(msg)
                raise ValueError(msg)

            for k, v in config:
                if v.endswith('.json'):
                    v = safe_json_load(v)

                self[k] = v
