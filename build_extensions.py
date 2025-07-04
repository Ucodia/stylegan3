#!/usr/bin/env python3
"""Utility script to precompile CUDA extensions used by StyleGAN3."""

import argparse
import importlib.machinery
import os
import shutil

from torch_utils import custom_ops
from torch_utils.ops import bias_act, filtered_lrelu, upfirdn2d


def _copy_module(module, name, outdir):
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        if module.__file__.endswith(suffix):
            dst = os.path.join(outdir, name + suffix)
            shutil.copyfile(module.__file__, dst)
            break


def build_extensions(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    os.environ['TORCH_EXTENSIONS_DIR'] = outdir
    custom_ops.verbosity = 'full'

    bias_act._plugin = None
    filtered_lrelu._plugin = None
    upfirdn2d._plugin = None

    bias_act._init()
    _copy_module(bias_act._plugin, 'bias_act_plugin', outdir)

    filtered_lrelu._init()
    _copy_module(filtered_lrelu._plugin, 'filtered_lrelu_plugin', outdir)

    upfirdn2d._init()
    _copy_module(upfirdn2d._plugin, 'upfirdn2d_plugin', outdir)

    print(f'Extensions built in {outdir}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Precompile StyleGAN3 CUDA extensions')
    parser.add_argument('--outdir', default='precompiled', help='Output directory for compiled extensions')
    args = parser.parse_args()
    build_extensions(os.path.abspath(args.outdir))


if __name__ == '__main__':
    main()

