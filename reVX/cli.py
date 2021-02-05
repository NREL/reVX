# -*- coding: utf-8 -*-
"""
reVX command line interface (CLI).
"""
import click
import logging
import os

from rex.utilities.cli_dtypes import STR, STRLIST, FLOAT
from rex.utilities.loggers import init_logger
from rex.utilities.utilities import safe_json_load

from reVX.utilities.exclusions_converter import ExclusionsConverter
from reVX.utilities.forecasts import Forecasts
from reVX.utilities.output_extractor import output_extractor
from reVX.utilities.region import RegionClassifier
from reVX.wind_setbacks.setbacks_converter import SetbacksConverter

from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """reVX command line interface."""
    ctx.ensure_object(dict)
    if verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'

    init_logger('reVX', log_level=log_level)


@main.command()
def version():
    """
    print version
    """
    click.echo(__version__)


@main.command()
@click.option('--meta_path', '-mp', required=True,
              prompt='Path to meta CSV file, resource .h5 file',
              type=click.Path(exists=True),
              help=('Path to meta CSV file, resource .h5 file containing '
                    'lat/lon points'))
@click.option('--regions_path', '-rp', required=True,
              prompt='Path to regions shapefile',
              type=click.Path(exists=True),
              help=('Path to regions shapefile containing labeled geometries'))
@click.option('--regions_label', '-rl', default=None, type=STR,
              show_default=True,
              help=('Attribute to use as label in the regions shapefile'))
@click.option('--fout', '-o', required=True,
              prompt='Output CSV file path',
              type=click.Path(),
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


@main.command()
@click.option('--my_fpath', '-src', required=True,
              type=click.Path(exists=True),
              help='Path to multi-year output .h5 file')
@click.option('--out_fpath', '-out', required=True,
              type=click.Path(),
              help='Path to output .h5 file')
@click.option('--year', '-yr', default=None, type=STR,
              show_default=True,
              help='Year to extract, if None parse from out_fpath')
def extract_output_year(my_fpath, out_fpath, year):
    """
    Extract all datasets for a give year from multi-year output file
    """
    output_extractor(my_fpath, out_fpath, year=year)


@main.command()
@click.option('--fcst_h5', '-fcst', required=True,
              type=click.Path(exists=True),
              help="Path to forecast .h5 file")
@click.option('--fcst_dset', '-fdset', required=True, type=str,
              help="Dataset to correct")
@click.option('--out_h5', '-out', required=True, type=click.Path(),
              help="Output path for corrected .h5 file")
@click.option('--actuals_h5', '-actuals', type=click.Path(exists=False),
              default=None, show_default=True,
              help="Path to forecast to .h5 file, by default None")
@click.option('--actuals_dset', '-adset', default=None, type=STR,
              show_default=True,
              help="Actuals dataset, by default None")
@click.option('--fcst_perc', '-perc', default=None, type=FLOAT,
              show_default=True,
              help=("Percentage of forecast to use for blending, by default "
                    "None"))
def correct_forecast(fcst_h5, fcst_dset, out_h5, actuals_h5, actuals_dset,
                     fcst_perc):
    """
    Bias correct and blend (if requested) forecasts using actuals:
    - Bias correct forecast data using bias correction factor:
    total actual generation / total forecasted generation
    - Blend fcst_perc of forecast generation with (1 - fcst_perc) of
    actuals generation
    """
    Forecasts.correct(fcst_h5, fcst_dset, out_h5, actuals_h5=actuals_h5,
                      actuals_dset=actuals_dset, fcst_perc=fcst_perc)


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
              help=(".json file containing mapping of layer names to geotiffs."
                    " Json can also contain layer descriptions and/or "
                    "scale factors"))
@click.option('-check_tiff', '-ct', is_flag=True,
              help=("Flag to check tiff profile and coordinates against "
                    "exclusion .h5 profile and coordinates"))
@click.option('-setbacks', '-sb', is_flag=True,
              help=("Flag to convert setbacks to exclusion layers"))
@click.option('--transform_atol', '-tatol', default=0.01, type=float,
              show_default=True,
              help=("Absolute tolerance parameter when comparing geotiff "
                    "transform data."))
@click.option('--coord_atol', '-catol', default=0.00001, type=float,
              show_default=True,
              help=("Absolute tolerance parameter when comparing new "
                    "un-projected geotiff coordinates against previous "
                    "coordinates."))
@click.option('--purge', '-r', is_flag=True,
              help="Remove existing .h5 file before loading layers")
@click.pass_context
def layers_to_h5(ctx, layers, check_tiff, setbacks, transform_atol, coord_atol,
                 purge):
    """
    Add layers to exclusions .h5 file
    """
    excl_h5 = ctx.obj['EXCL_H5']
    if purge and os.path.isfile(excl_h5):
        os.remove(excl_h5)

    inputs = safe_json_load(layers)
    layers = inputs['layers']
    descriptions = inputs.get('descriptions')
    scale_factors = inputs.get('scale_factors')

    if setbacks:
        SetbacksConverter.layers_to_h5(excl_h5, layers,
                                       check_tiff=check_tiff,
                                       transform_atol=transform_atol,
                                       coord_atol=coord_atol,
                                       descriptions=descriptions,
                                       scale_factors=scale_factors)
    else:
        ExclusionsConverter.layers_to_h5(excl_h5, layers,
                                         check_tiff=check_tiff,
                                         transform_atol=transform_atol,
                                         coord_atol=coord_atol,
                                         descriptions=descriptions,
                                         scale_factors=scale_factors)


@exclusions.command()
@click.option('--out_dir', '-o', required=True, type=click.Path(exists=True),
              help=("Output directory to save layers into"))
@click.option('--layers', '-l', default=None, type=STRLIST,
              show_default=True,
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
        layers = {layer: os.path.join(out_dir, "{}.tif".format(layer))
                  for layer in layers}
        ExclusionsConverter.extract_layers(excl_h5, layers, hsds=hsds)
    else:
        ExclusionsConverter.extract_all_layers(excl_h5, out_dir, hsds=hsds)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reVX CLI')
        raise
