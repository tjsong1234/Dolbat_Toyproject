"""Small clamped B-spline basis and analytic derivatives, using NumPy only."""
import numpy as np


def clamped_knots(domain_cm, control_points=7, degree=3):
    lo,hi=map(float,domain_cm)
    if not np.isfinite([lo,hi]).all() or hi<=lo:
        raise ValueError('Finite increasing spline domain required')
    if not isinstance(control_points,(int,np.integer)) or control_points<degree+1 or degree<0:
        raise ValueError('Enough control points for the spline degree required')
    interior=control_points-degree-1
    middle=np.linspace(lo,hi,interior+2)[1:-1] if interior else np.array([])
    return np.r_[np.repeat(lo,degree+1),middle,np.repeat(hi,degree+1)]


def _basis_values(parameter_cm, knots, degree, derivatives):
    """Evaluate requested derivatives with one shared lower-degree memo.

    Values outside the clamped domain are zero. Callers doing geometric fitting
    must constrain parameters to the actual model domain rather than silently
    treating these zeros as extrapolation.
    """
    x=np.asarray(parameter_cm,dtype=float).reshape(-1)
    k=np.asarray(knots,dtype=float)
    if (k.ndim!=1 or len(k)<2*(degree+1) or degree<0 or any(d<0 for d in derivatives)
            or not np.isfinite(k).all() or np.any(np.diff(k)<0) or not np.isfinite(x).all()):
        raise ValueError('Finite ordered knots, parameters and nonnegative derivative required')
    lo,hi=k[degree],k[-degree-1]
    inside=(x>=lo)&(x<=hi)
    xx=x.copy();xx[x==hi]=np.nextafter(hi,lo)
    memo={}
    def evaluate(p,d):
        key=(p,d)
        if key in memo:return memo[key]
        n=len(k)-p-1
        if d>p:result=np.zeros((len(x),n))
        elif d:
            lower=evaluate(p-1,d-1)
            left=k[p:p+n]-k[:n];right=k[p+1:p+1+n]-k[1:1+n]
            a=np.divide(p,left,out=np.zeros(n),where=left>0)
            b=np.divide(p,right,out=np.zeros(n),where=right>0)
            result=lower[:,:n]*a-lower[:,1:n+1]*b
        elif p==0:
            result=((xx[:,None]>=k[:-1])&(xx[:,None]<k[1:])).astype(float)
        else:
            lower=evaluate(p-1,0)
            left=k[p:p+n]-k[:n];right=k[p+1:p+1+n]-k[1:1+n]
            a=np.divide(xx[:,None]-k[:n],left,out=np.zeros((len(x),n)),where=left>0)
            b=np.divide(k[p+1:p+1+n]-xx[:,None],right,out=np.zeros((len(x),n)),where=right>0)
            result=a*lower[:,:n]+b*lower[:,1:n+1]
        result[~inside]=0.
        memo[key]=result
        return result
    result=tuple(evaluate(degree,d) for d in derivatives)
    # A recursive nested function otherwise keeps its own closure (and every
    # temporary basis array in memo) alive until cyclic garbage collection.
    # The requested matrices are complete, so break that self-reference now.
    evaluate=None
    return result


def bspline_basis(parameter_cm, knots, degree=3, derivative=0):
    """Return N x controls basis values; endpoint derivatives are left limits.

    Values outside the clamped domain are zero. No input or knot is modified.
    """
    return _basis_values(parameter_cm,knots,degree,(derivative,))[0]


def bspline_basis_many(parameter_cm, knots, degree=3, derivatives=(0,1,2)):
    """Return several derivative matrices without repeating their recurrence.

    The arithmetic of each matrix matches :func:`bspline_basis`; only common
    lower-degree values and validation work are reused within this call.
    """
    return _basis_values(parameter_cm,knots,degree,tuple(derivatives))


def basis_matrix(parameter_cm, knots, derivative=0):
    """Convenience cubic-basis alias."""
    return bspline_basis(parameter_cm,knots,3,derivative)
