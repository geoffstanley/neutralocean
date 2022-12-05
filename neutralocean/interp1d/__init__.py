"""
Functions for interpolation of one (or two) dependent variables in terms of one
independent variable, done serially over an arbitrary number of such 
interpolation problems.  
Unlike `ppinterp`, the entire interpolant is not calculated: only the minimum
amount of data near the evaluation site is accessed, and the interpolant is 
evaluated immediately.  Thus, if many interpolations are going to be performed,
this approach is slower than `ppinterp`, but it has a lower memory footprint,
so this is the generally recommended method.
"""
from .tools import make_interpolator, make_kernel
