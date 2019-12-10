# -*- coding: utf-8 -*-
"""
reVX command line interface (CLI).
"""
import click
import logging
from reV.utilities.cli_dtypes import STR
from reVX.utilities.region import RegionClassifier

logger = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """reVX command line interface."""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--meta_path', '-i', required=True,
              prompt='Path to meta CSV file',
              type=click.Path(exists=True),
              help=('Path to meta CSV file containing lat/lon points'))
@click.option('--regions_path', '-r', required=True,
              prompt='Path to regions shapefile',
              type=click.Path(exists=True),
              help=('Path to regions shapefile containing labeled geometries'))
@click.option('--regions_label', '-l', default=None, type=STR,
              help=('Attribute to use as label in the regions shapefile'))
@click.option('--fout', '-o', required=True,
              prompt='Output CSV file path',
              type=click.Path(exists=False),
              help=('Output CSV file path for labeled meta CSV file'))
@click.option('--force', '-f', is_flag=True,
              help='Force outlier classification by finding nearest.')
def region_classifier(meta_path, regions_path, regions_label, fout, force):
    """
    Region Classifier
    - Used to classify meta points with a label from a shapefile
    """

    RegionClassifier.run(meta_path=meta_path,
                         regions_path=regions_path,
                         regions_label=regions_label,
                         force=force, fout=fout)


if __name__ == '__main__':
    main(obj={})
