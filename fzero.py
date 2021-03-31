def brent(f, a, b, t):
    #FZERO_BRENT  Find a root of a univariate function within a given interval
    #             using Brent's method
    #
    # x = fzero_brent(f,a,b,t)
    # finds x in the interval [a,b] satisfying |x-y| <= t/2 where f(y) = 0.
    # f(a) and f(b) must have opposite signs.
    #
    # ... = fzero_brent(f,lb,ub,t,x,...) passes additional inputs directly
    # to f.
    #
    # This function is compatible with MATLAB's code generation -- so long f is
    # similarly compatible. Many root-finding problems can be solved with
    # fzero_brent by writing another function which calls fzero_brent inside a
    # for loop, and this can be made fast by using code generation on that
    # wrapper function. Note that f will only be called with a scalar input as
    # its first argument; codegen knows this, and might strip out unnecessary
    # code from the function definition underlying f.
    # --- Example:
    # Simple bisection between bounds of -0.5 and 1.5 would fail to find the
    # root. Starting with a guess of .85 and expanding outwards finds a root.
    # This example is shown graphically on the MATLAB File Exchange.
    #
    # c = poly([0 1]); # a polynomial with roots at 0 and 1
    # f = @(x) polyval(c,x); # encapsulate extra parameters
    # [a,b] = fzero_guess_to_bounds(f, .85, -.5, 1.5);
    # root = fzero_brent(f, a, b, .05);
    #
    #  Discussion:
    #
    #    The interval [A,B] must be a change of sign interval for F.
    #    That is, F(A) and F(B) must be of opposite signs.  Then
    #    assuming that F is continuous implies the existence of at least
    #    one value C between A and B for which F(C) = 0.
    #
    #    The location of the zero is determined to within an accuracy
    #    of 6 * EPS * abs ( C ) + 2 * T, where EPS is the machine epsilon.
    #
    #    Thanks to Thomas Secretin for pointing out a transcription error in the
    #    setting of the value of P, 11 February 2013.
    #
    #    Additional parameters given by varargin are passed directly to F.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    14 September 2019
    #    30 June 2020 - Geoff Stanley
    #
    #  Author:
    #
    #    Original FORTRAN77 version by Richard Brent.
    #    MATLAB version by John Burkardt.
    #    Minor changes (passing extra arguments) by Geoff Stanley.
    #
    #  Reference:
    #
    #    Richard Brent,
    #    Algorithms for Minimization Without Derivatives,
    #    Dover, 2002,
    #    ISBN: 0-486-41998-3,
    #    LC: QA402.5.B74.
    #
    #  Parameters:
    #
    #    Input, real A, B, the endpoints of the change of sign interval.
    #
    #    Input, real T, a positive error tolerance.
    #
    #    Input, real value = F ( x ), the name of a user-supplied
    #    function which evaluates the function whose zero is being sought.
    #
    #    Output, real VALUE, the estimated value of a zero of
    #    the function F.

    fa = f(a)
    fb = f(b)

    c = a
    fc = fa
    e = b - a
    d = e
    ϵ = 7./3 - 4./3 - 1 # == 2.220446049250313e-16 # assuming float64 inputs

    while True:

        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol = 2. * ϵ * abs(b) + t
        m = 0.5 * ( c - b )

        if abs(m ) <= tol or fb == 0.:
            break

        if abs(e) < tol or abs(fa) <= abs(fb):
            e = m
            d = e
        else:
            s = fb / fa
            if a == c:
                p = 2. * m * s
                q = 1. - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * ( 2. * m * q * ( q - r ) - ( b - a ) * ( r - 1. ) )
                q = ( q - 1. ) * ( r - 1. ) * ( s - 1. )

            if 0. < p:
                q = -q
            else:
                p = -p

            s = e
            e = d

            if 2. * p < 3. * m * q - abs(tol * q) and p < abs(0.5 * s * q):
                d = p / q
            else:
                e = m
                d = e

        a = b
        fa = fb

        if tol < abs(d):
            b += d
        elif 0. < m:
            b += tol
        else:
            b -= tol


        fb = f(b)

        if ( 0. < fb and 0. < fc ) or ( fb <= 0. and fc <= 0. ):
            c = a
            fc = fa
            e = b - a
            d = e


    return b



def guess_to_bounds(f, x, lb, ub):
    #FZERO_GUESS_TO_BOUNDS  Search for a sign change bounding a zero of a
    #                       univariate function, expanding geometrically
    #                       outward from an initial guess.
    #
    #
    # [a, b] = fzero_guess_to_bounds(f, x)
    # finds a < b such that f(a) and f(b) have different sign*, meaning a
    # solution exists within the interval [a,b].  The bounds a,b are expanded
    # outward in geometric progression from an initial guess for the root of f
    # at x. If f evaluates to NaN at any point during the search, then a = nan
    # and b = nan are immediately returned.  If the function is genuinely
    # single-signed, or even if it is not but its values of opposite sign are
    # skipped over, it is possible to enter an infinite loop.  Calling the
    # function in this form is therefore not recommended unless you know the
    # function will not result in such an infinite loop.
    #
    # [a, b] = fzero_guess_to_bounds(f, x, lb, ub)
    # as above, but limits [a,b] to remain inside the subset [lb, ub].  If x is
    # outside of [lb, ub], it is immediately moved into this range. If no
    # sign-change is found within [lb, ub], then a = nan and b = nan are
    # returned.  Note, as above, it is possible that a sign-change is skipped
    # over as f is only evaluated at finitely many x values.
    #
    # [a,b] = fzero_guess_to_bounds(f, x, lb, ub, ...)
    # passes all additional arguments to the function f.
    #
    # * Note: for computational speed, herein the "sign" of 0 is considered the
    # same as the sign of a negative number.
    #
    # This function is compatible with MATLAB's code generation.
    #
    #
    # --- Input:
    #   f       : handle to a function that accepts a real scalar as its first
    #             input and returns a real scalar
    #   x       : initial scalar guess for a root of f
    #   lb      : scalar lower bound
    #   ub      : scalar upper bound
    #   varargin: All additional inputs are passed directly to f
    #
    #
    # --- Output:
    #   a : lower bound for interval containing a root, scalar
    #   b : upper bound for interval containing a root, scalar
    #
    #
    # --- Acknowledgements:
    # Expansion from initial guess inspired by MATLAB's fzero.m.
    #
    #
    # Author    : Geoff Stanley
    # Email     : geoffstanley@gmail.com
    # Version   : 1.0
    # History   : 01/07/2020 - initial release
    #           : 22/07/2020 - fix infinite loop in bounded case, arising from machine precision rounding

    # Geometrically expand from the guess x; until a sign change is found

    nan = float("nan")

    x = min(max(x, lb), ub)

    # bounds are given
    dxp = (ub - x) / 50
    dxm = (x - lb) / 50

    # Set a = x; except when x is so close to lb that machine roundoff makes dxm identically 0
    # which would lead to an infinite loop below.  In this case; set a = lb.
    if dxm == 0:
        a = lb
    else:
        a = x

    # Similarly; set b = x; except for machine precision problems.
    if dxp == 0:
        b = ub
    else:
        b = x

    fapos = f(a) > 0.
    fbpos = f(b) > 0.

    while True:
        if a > lb:
            # Move a left; & test for a sign change
            dxm *= 1.414213562373095
            a = max(x - dxm, lb)
            fapos = f(a) > 0.
            if fapos ^ fbpos: # fa & fb have different signs
                return (a,b)
        elif b == ub: # also a .== lb; so cannot expand anymore
            if fapos ^ fbpos: # one last test for sign change
                return (a,b)
            else: # no sign change found
                return (nan,nan)

        if b < ub:
            # Move b right; & test for a sign change
            dxp *= 1.414213562373095
            b = min(x + dxp, ub)
            fbpos = f(b) > 0.
            if fapos ^ fbpos: # fa & fb have different signs
                return (a,b)
        elif a == lb: # also b .== ub; so cannot expand anymore
            if fapos ^ fbpos: # one last test for sign change
                return (a,b)
            else: # no sign change found
                return (nan,nan)


## Tests:
# myfun_multivar = lambda x,p: (x-2-p) * (x+3-p)  # roots at 2+p and -3+p
# myfun_univar = lambda x: myfun_multivar(x, 0.5)  # roots at 2.5 and -2.5
# (a,b) = guess_to_bounds(myfun_univar, -1., -6., 0.)
# brent(myfun_univar, a, b, 1e-4)
