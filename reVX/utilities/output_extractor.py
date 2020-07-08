# -*- coding: utf-8 -*-
"""
Utility to extract a years data from multi-year output .h5 files
"""
import h5py
import logging
import os

from rex.utilities.utilities import parse_year

logger = logging.getLogger(__name__)


def output_extractor(my_fpath, out_fpath, year=None):
    """
    Extract all datasets for a give year from multi-year output file

    Parameters
    ----------
    my_fpath : str
        Path to multi-year output .h5 file
    out_fpath : str
        Path to output .h5 file
    year : int | str, optional
        Year to extract, if None parse from out_fpath, by default None
    """
    if year is None:
        year = parse_year(os.path.basename(out_fpath))

    if not isinstance(year, str):
        year = str(year)

    logger.info('Transfering all datasets for {} from {} to {}'
                .format(year, my_fpath, out_fpath))
    with h5py.File(out_fpath, 'w-') as f_out:
        with h5py.File(my_fpath, 'r') as f_src:
            logger.debug('Transfering global attrs')
            for k, v in f_src.attrs.items():
                f_out.attrs[k] = v

            logger.debug('Transfering meta')
            src = f_src['meta']
            out = f_out.create_dataset('meta', shape=src.shape,
                                       dtype=src.dtype,
                                       chunks=src.chunks,
                                       data=src[...])
            for k, v in src.attrs.items():
                out.attrs[k] = v

            for dset in f_src:
                if dset.endswith(year):
                    out_name = dset.rstrip('-{}'.format(year))
                    logger.debug('Transfering {}'.format(out_name))
                    src = f_src[dset]
                    out = f_out.create_dataset(out_name,
                                               shape=src.shape,
                                               dtype=src.dtype,
                                               chunks=src.chunks,
                                               data=src[...])
                    for k, v in src.attrs.items():
                        out.attrs[k] = v

    logger.info('{} created successfully'.format(out_fpath))
