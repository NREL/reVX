# -*- coding: utf-8 -*-
"""
reVX command line interface (CLI).
"""
import click


@click.group()
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """reVX command line interface."""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


if __name__ == '__main__':
    main(obj={})
