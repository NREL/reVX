"""
Classification Command Line Interface
"""

import click
import logging
from reV.utilities.cli_dtypes import STR
from reVX.classification.region import region_classifier

logger = logging.getLogger(__name__)


@click.command()
@click.option('--meta_path', '-i', required=True,
              prompt='Path to meta CSV file',
              type=click.Path(exists=True),
              help=('Path to meta CSV file containing lat/lon points'))
@click.option('--regions_path', '-r', required=True,
              prompt='Path to regions shapefile',
              type=click.Path(exists=True),
              help=('Path to regions shapefile containing labeled geometries'))
@click.option('--regions_label', '-l', default=None, type=STR,
              help=('Attribute to use a label in the regions shapefile'))
@click.option('--lat_label', '-y', default='LATITUDE', type=STR,
              help=('Latitude column name in meta CSV file'))
@click.option('--long_label', '-x', default='LONGITUDE', type=STR,
              help=('Longitude column name in meta CSV file'))
@click.option('--save_to', '-o', required=True,
              prompt='Output CSV file path',
              type=click.Path(exists=False),
              help=('Output CSV file path for labeled meta CSV file'))
@click.option('--force', '-f', is_flag=True,
              help='Force outlier classification by finding nearest.')
def main(meta_path, regions_path, lat_label, long_label, regions_label,
         save_to, force):
    """
    Region Classifier
    - Used to classify meta points with a label from a shapefile
    """

    classifier = region_classifier(meta_path=meta_path,
                                   regions_path=regions_path,
                                   lat_label=lat_label, long_label=long_label,
                                   regions_label=regions_label)

    classifier.classify(save_to=save_to, force=force)
