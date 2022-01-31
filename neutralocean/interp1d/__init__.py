"""
Functions for linear or PCHIP interpolation of one (or two) dependent variables
in terms of one independent variable, done serially over an arbitrary number
of such interpolation problems.  This is used to interpolate salinity and
temperature in terms of either pressure or depth in each water column. 
"""
from .tools import make_interpolator, make_kernel
