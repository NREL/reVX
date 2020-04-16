# -*- coding: utf-8 -*-
"""
reVX command line interface (CLI).
"""
import click
import logging
import os

from rex.utilities.cli_dtypes import STR, STRLIST
from rex.utilities.utilities import safe_json_load

from reVX.utilities.exclusions_converter import ExclusionsConverter
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
def region_classifier(meta_path, regions_path, regions_label, fout,
                      force):
    """
    Region Classifier
    - Used to classify meta points with a label from a shapefile
    """
    RegionClassifier.run(meta_path=meta_path,
                         regions_path=regions_path,
                         regions_label=regions_label,
                         force=force, fout=fout)


@main.group()
@click.option('--excl_h5', '-h5', required=True, type=click.Path(exists=False),
              help=("Path to .h5 file containing or to contain exclusion "
                    "layers"))
@click.pass_context
def exclusions(ctx, excl_h5):
    """
    Extract from or create exclusions .h5 file
    """
    ctx.obj['EXCL_H5'] = excl_h5


@exclusions.command()
@click.option('--layers', '-l', required=True, type=click.Path(exists=True),
              help=(".json file containing list of geotiffs to load or "
                    "mapping of layer names to geotiffs"))
@click.option('--descriptions', '-d', default=None,
              type=click.Path(exists=True),
              help=(".json file containing layer descriptions as a list or "
                    "mapping to layers"))
@click.option('--transform_atol', '-tatol', default=0.01, type=float,
              help=("Absolute tolerance parameter when comparing geotiff "
                    "transform data."))
@click.option('--coord_atol', '-catol', default=0.00001, type=float,
              help=("Absolute tolerance parameter when comparing new "
                    "un-projected geotiff coordinates against previous "
                    "coordinates."))
@click.option('--purge', '-r', is_flag=True,
              help="Remove existing .h5 file before loading layers")
@click.pass_context
def layers_to_h5(ctx, layers, descriptions, transform_atol, coord_atol, purge):
    """
    Add layers to exclusions .h5 file
    """
    excl_h5 = ctx.obj['EXCL_H5']
    if purge and os.path.isfile(excl_h5):
        os.remove(excl_h5)

    layers = safe_json_load(layers)
    if 'layers' in layers:
        layers = layers['layers']

    if descriptions is not None:
        descriptions = safe_json_load(descriptions)
        if 'descriptions' in descriptions:
            descriptions = {os.path.basename(l).split('.')[0]: d
                            for l, d in zip(layers, descriptions)}

    ExclusionsConverter.layers_to_h5(excl_h5, layers,
                                     transform_atol=transform_atol,
                                     coord_atol=coord_atol,
                                     descriptions=descriptions)


@exclusions.command()
@click.option('--out_dir', '-o', required=True, type=click.Path(exists=True),
              help=("Output directory to save layers into"))
@click.option('--layers', '-l', default=None, type=STRLIST,
              help=("List of layers to extract, if None extract all"))
@click.option('--hsds', '-hsds', is_flag=True,
              help="Extract layers from HSDS")
@click.pass_context
def layers_from_h5(ctx, out_dir, layers, hsds):
    """
    Extract layers from excl .h5 file and save to disk as geotiffs
    """
    excl_h5 = ctx.obj['EXCL_H5']
    if layers is not None:
        layers = {l: os.path.join(out_dir, "{}.tif".format(l)) for l in layers}
        ExclusionsConverter.extract_layers(excl_h5, layers, hsds=hsds)
    else:
        ExclusionsConverter.extract_all_layers(excl_h5, out_dir, hsds=hsds)


if __name__ == '__main__':
    main(obj={})
