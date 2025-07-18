# -*- coding: utf-8 -*-
"""
reVX command line interface (CLI).
"""
import click
import logging
import os
import json
from pathlib import Path
import pandas as pd

from gaps.config import load_config
from gaps.cli import CLICommandFromFunction, as_click_command

from rex.utilities.cli_dtypes import STR, STRLIST, FLOAT
from rex.utilities.loggers import init_logger
from rex.utilities.utilities import safe_json_load
from reV.supply_curve.exclusions import ExclusionMaskFromDict, ExclusionLayers

from reVX.offshore.dist_to_ports_converter import DistToPortsConverter
from reVX.handlers.geotiff import Geotiff
from reVX.handlers.layered_h5 import LayeredH5
from reVX.utilities.forecasts import FcstUtils
from reVX.utilities.output_extractor import output_extractor
from reVX.utilities.region_classifier import RegionClassifier
from reVX.utilities.reeds_cols import add_reeds_columns
from reVX.setbacks.setbacks_converter import SetbacksConverter
from reVX.utilities.fix_sc_lat_lons import fix_sc_lat_lon
from reVX.utilities.utilities import rev_sc_to_geotiff_arr

from reVX import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
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
                         regions=regions_path,
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
    FcstUtils.correct(fcst_h5, fcst_dset, out_h5, actuals_h5=actuals_h5,
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
              help=("Flag to check tiff profile, CRS, and shape against "
                    "exclusion .h5 profile, CRS, and shape"))
@click.option('--setbacks', '-sb', is_flag=True,
              help=("Flag to convert setbacks to exclusion layers"))
@click.option('--distance_to_ports', '-dtp', is_flag=True,
              help=("Flag to convert distances to ports to exclusion layers"))
@click.option('--transform_atol', '-tatol', default=0.01, type=float,
              show_default=True,
              help=("Absolute tolerance parameter when comparing geotiff "
                    "transform data."))
@click.option('--purge', '-r', is_flag=True,
              help="Remove existing .h5 file before loading layers")
@click.pass_context
def layers_to_h5(ctx, layers, check_tiff, setbacks, distance_to_ports,
                 transform_atol, purge):
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

    if setbacks and distance_to_ports:
        msg = ('Both the setbacks "--setbacks/-sb" and distance to ports '
               '"--distance_to_ports/-dtp" flags were supplied! Setbacks and '
               'distance to ports and computed differently, please only '
               'select one!')
        logger.error(msg)
        raise RuntimeError(msg)

    if setbacks:
        are_inclusion_layers = inputs.get('are_inclusion_layers', False)
        converter = SetbacksConverter(excl_h5)
        converter.layers_to_h5(layers, check_tiff=check_tiff,
                               are_inclusion_layers=are_inclusion_layers,
                               transform_atol=transform_atol,
                               descriptions=descriptions,
                               scale_factors=scale_factors)
    elif distance_to_ports:
        converter = DistToPortsConverter(excl_h5)
        converter.layers_to_h5(layers, check_tiff=check_tiff,
                               transform_atol=transform_atol,
                               descriptions=descriptions,
                               scale_factors=scale_factors)
    else:
        converter = LayeredH5(excl_h5)
        converter.layers_to_h5(layers, check_tiff=check_tiff,
                               transform_atol=transform_atol,
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
        LayeredH5(excl_h5, hsds=hsds).extract_layers(layers)
    else:
        LayeredH5(excl_h5, hsds=hsds).extract_all_layers(out_dir)


@exclusions.command()
@click.option('--excl_dict_fpath', '-ed', required=True,
              type=click.Path(exists=True),
              help=('Path to JSON file containing the ``"excl_dict"`` '
                    'key which points to the exclusion dictionary defining '
                    'the mask that should be generated. A typical reV '
                    'aggregation config satisfied this requirement. If this '
                    'file also contains an ``"excl_fpath"`` key, the value '
                    'from the file will override the ``--excl_h5`` CLI '
                    'argument input.'))
@click.option('--out', '-o', required=True, type=STR,
              help=('Output name. If this string value ends in ".tif" '
                    'or ".tiff", this input is assumed to be a path to an '
                    'output tiff file, and the mask will be written to that '
                    'destination. Otherwise, this input is assumed to be the '
                    'name of the layer in the exclusion file to write the '
                    'mask to.'))
@click.option('--min_area', '-ma', default=None, type=FLOAT,
              help=('Minimum required contiguous area in sq-km.'))
@click.option('--kernel', '-k', type=STR, default='queen',
              show_default=True,
              help=('Contiguous filter method to use on final exclusion.'))
@click.option('--hsds', '-hsds', is_flag=True,
              help=('Flag to use h5pyd to handle .h5 domain hosted on AWS '
                    'behind HSDS'))
@click.pass_context
def mask(ctx, excl_dict_fpath, out, min_area, kernel, hsds):
    """
    Compute Setbacks locally
    """
    log_level = "DEBUG" if ctx.obj.get('VERBOSE') else "INFO"
    init_logger('reV', log_level=log_level)
    init_logger('reVX', log_level=log_level)

    logger.info("Calculating exclusion mask from {!r}".format(excl_dict_fpath))

    with open(excl_dict_fpath, 'r') as fh:
        config = json.load(fh)

    excl_fpath = config.get("excl_fpath", ctx.obj['EXCL_H5'])
    logger.debug("Exclusion filepath(s): {!r}".format(excl_fpath))

    excl_dict = config['excl_dict']
    logger.debug("Exclusion dictionary: {!r}".format(excl_dict))
    mask_ = ExclusionMaskFromDict.run(excl_fpath, layers_dict=excl_dict,
                                      min_area=min_area, kernel=kernel,
                                      hsds=hsds)

    with ExclusionLayers(excl_fpath, hsds=hsds) as f:
        profile = f.profile

    if out.endswith(".tif") or out.endswith(".tiff"):
        out = Path(out).resolve().as_posix()
        logger.info("Writing mask to {!r}".format(out))
        Geotiff.write(out, profile, mask_)
    else:
        if isinstance(excl_fpath, list):
            excl_fpath = excl_fpath[0]
        logger.info("Writing mask to layer {!r} in {!r}"
                    .format(out, excl_fpath))
        desc = ("Exclusion mask computed from exclusion dictionary: {!r}"
                .format(excl_dict))
        LayeredH5(excl_fpath).write_layer_to_h5(mask_, out, profile,
                                                description=desc)

@main.command()
@click.option('--agg_config', '-ac', required=True,
              type=click.Path(exists=True),
              help=('Path to config file containing the ``"excl_fpath"`` '
                    'key which points to the exclusion file(s) used for the '
                    'reV aggregation run. A typical reV aggregation config '
                    'satisfied this requirement.'))
@click.argument('sc_fps', type=click.Path(exists=True), nargs=-1)
def fix_rev_sc_lat_lon(agg_config, sc_fps):
    """Fix reV SC lat/lon values (in-place) using pyproj"""
    config = load_config(agg_config)
    fixed_scs = fix_sc_lat_lon(sc_fps, excl_fp=config["excl_fpath"],
                               resolution=config["resolution"], as_gpkg=False)

    for sc_fp, fixed_sc in fixed_scs.items():
        fixed_sc.to_csv(sc_fp, index=False)


@main.command()
@click.option('--agg_config', '-ac', required=True,
              type=click.Path(exists=True),
              help=('Path to config file containing the ``"excl_fpath"`` '
                    'key which points to the exclusion file(s) used for the '
                    'reV aggregation run. A typical reV aggregation config '
                    'satisfied this requirement.'))
@click.argument('sc_fps', type=click.Path(exists=True), nargs=-1)
def rev_sc_to_gpkg(agg_config, sc_fps):
    """
    Convert reV SC to a GeoPackage where the geometries are supply
    curve cells
    """
    config = load_config(agg_config)
    gpkgs = fix_sc_lat_lon(sc_fps, excl_fp=config["excl_fpath"],
                           resolution=config["resolution"], as_gpkg=True)

    for sc_fp, gpkg in gpkgs.items():
        out_fp = sc_fp.replace(".csv", ".gpkg")
        gpkg.to_file(out_fp, driver="GPKG", index=False)


@main.command()
@click.option('--agg_config', '-ac', required=True,
              type=click.Path(exists=True),
              help=('Path to config file containing the ``"excl_fpath"`` '
                    'key which points to the exclusion file(s) used for the '
                    'reV aggregation run. A typical reV aggregation config '
                    'satisfied this requirement.'))
@click.option('--sc_fp', '-sc', type=click.Path(exists=True), required=True,
              help=('Path to reV supply curve'))
@click.argument('cols', type=STR, nargs=-1)
def rev_sc_to_tiff(agg_config, sc_fp, cols):
    """
    Convert reV SC to a GeoTiff containing data from the specified columns
    """
    config = load_config(agg_config)
    sc = pd.read_csv(sc_fp)

    sc_tiffs = rev_sc_to_geotiff_arr(sc, config["excl_fpath"],
                                     config["resolution"], cols,
                                     dtype="float32")

    for col, value_array, profile in sc_tiffs:
        out_fp = sc_fp.replace(".csv", f"_{col}.tif")
        Geotiff.write(out_fp, profile, value_array, dtype="float32")


def _reeds_cols_preprocessor(config):
    """Ensure supply_curve_fpath and out_fpath inputs are lists """
    for key in ["supply_curve_fpath", "out_fp"]:
        if isinstance(config.get(key), str):
            config[key] = [config[key]]

    if config.get("out_fp") is None:
        config["out_fp"] = config["supply_curve_fpath"]

    if len(config["supply_curve_fpath"]) != len(config["out_fp"]):
        raise ValueError("Length of 'out_fp' must match length of "
                         "'supply_curve_fpath'")

    return config


command = CLICommandFromFunction(add_reeds_columns,
                                 name="add-reeds-cols",
                                 split_keys=[("supply_curve_fpath", "out_fp")],
                                 config_preprocessor=_reeds_cols_preprocessor)
main.add_command(as_click_command(command))


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reVX CLI')
        raise
