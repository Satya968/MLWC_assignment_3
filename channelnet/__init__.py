"""ChannelNet reproduction package.

A compact, self-contained implementation of deep-learning OFDM channel
estimation (SRCNN super-resolution + DnCNN denoising) together with the
classical LS-interpolation and MMSE baselines it is compared against.

The sub-modules are organised by responsibility:

    settings   experiment-wide constants (grid, splits, SNRs, pilots, paths)
    channels   read the MATLAB-generated perfect channels and split them
    pilots     lattice pilot grids, AWGN, least-squares at the pilots
    classical  LS + 2-D interpolation and the two MMSE variants
    imaging    complex channel  <->  real/imag single-channel images
    nets       the SRCNN and DnCNN torch modules

Nothing here touches the filesystem on import; paths come from `settings`.
"""

from . import settings  # noqa: F401
