"""
CLI to create wet (TODO and dry) costs, barriers, and friction layers. Final
layers required for LCP routing are saved to an H5 file. All layers may
optionally be saved to GeoTIFF.
"""

import click
import logging
logger = logging.getLogger(__name__)

from rex.utilities.loggers import init_logger, LOG_LEVEL

from reVX import __version__

@click.group()
@click.version_option(version=__version__)
@click.option('--log-level', type=click.Choice(LOG_LEVEL.keys()),
              default='INFO', help='Logging level.')
def main(log_level):
    """
    Transmission LCP routing cost, friction, and barrier layer creator.
    """
    init_logger('reVX', log_level=log_level)


@main.command
@click.option('-c', '--config', 'config_fpath', type=click.Path(exists=True),
              required=True, help='Configuration JSON.')
@click.option('--create-h5/--use-existing-h5', default=True, show_default=True,
              help='Create a new H5 data file or use an existing H5.')
def from_config(config_fpath: str, create_h5: bool):
    """
    Create costs, barriers, and frictions from a config file.
    """
    config = OffshoreCreatorConfig(config_fpath)


if __name__ == '__main__':
    try:
        main()  # pylint: disable=no-value-for-parameter
    except Exception:
        logger.exception('Error running Offshore Layer Creator CLI')
        raise