"""A joint centre/normal-width model fitted to current real boundary rows.

The spline is a geometric lane estimate, not a vehicle trajectory or measured
rear-axle pose. Generated opposite boundaries are outputs, never extra evidence.
"""
from dataclasses import dataclass,field
import numpy as np
from .bspline_basis import clamped_knots,bspline_basis,bspline_basis_many


@dataclass
class JointLaneModel:
    coefficients: np.ndarray
    knots: np.ndarray
    domain_cm: tuple
    width_mid_cm: float
    width_slope_cm_per_cm: float
    accepted: bool=False
    reason: str='unfitted'
    diagnostics: dict=field(default_factory=dict)
    evidence_weights: list=field(default_factory=list)

    def _parameter(self,t):
        t=np.asarray(t,dtype=float).reshape(-1)
        if not np.isfinite(t).all() or np.any(t<self.domain_cm[0]-1e-6) or np.any(t>self.domain_cm[1]+1e-6):
            raise ValueError('Parameter outside the current joint-model domain')
        return np.clip(t,*self.domain_cm)

    def _y(self,t,derivative=0):
        # Boundaries, centre and curvature use identical parameter grids. Keep
        # only four grids per short-lived model; coefficients are never cached.
        # Value keys also invalidate safely after in-place parameter/knot edits.
        key=(np.asarray(t,dtype=float).tobytes(),self.knots.tobytes())
        cache=getattr(self,'_basis_cache',None)
        if cache is None:self._basis_cache=cache={}
        if key not in cache:
            if len(cache)>=4:cache.pop(next(iter(cache)))
            cache[key]=dict(enumerate(bspline_basis_many(t,self.knots)))
        bases=cache[key]
        if derivative not in bases:bases[derivative]=bspline_basis(t,self.knots,3,derivative)
        return bases[derivative]@self.coefficients

    def center(self,t):
        t=self._parameter(t)
        return np.column_stack((t,self._y(t)))

    def slope(self,t):return self._y(self._parameter(t),1)

    def curvature(self,t):
        t=self._parameter(t);slope=self._y(t,1)
        return self._y(t,2)/(1+slope*slope)**1.5

    def width(self,t):
        t=self._parameter(t)
        return self.width_mid_cm+self.width_slope_cm_per_cm*(t-np.mean(self.domain_cm))

    def boundary(self,side,t):
        if side not in ('left','right'):raise ValueError('Boundary side must be left or right')
        t=self._parameter(t);slope=self._y(t,1);q=np.sqrt(1+slope*slope)
        distance=(.5 if side=='left' else -.5)*self.width(t)
        return np.column_stack((t-distance*slope/q,self._y(t)+distance/q))


def _project_center(points,coefficients,knots,domain,initial=None,max_iterations=8,tolerance_cm=1e-5,return_basis=False):
    """Closest normal correspondences; endpoint tangential residual stays explicit."""
    lo,hi=domain;x,y=points.T
    t=np.clip(x if initial is None else initial,lo,hi).copy()
    for _ in range(max_iterations):
        b,d,dd=bspline_basis_many(t,knots)
        f=b@coefficients;m=d@coefficients;second=dd@coefficients
        numerator=t-x+(f-y)*m;denominator=1+m*m+(f-y)*second
        step=np.divide(numerator,denominator,out=np.zeros_like(t),where=abs(denominator)>1e-6)
        updated=np.clip(t-np.clip(step,-(hi-lo)/5,(hi-lo)/5),lo,hi)
        change=float(np.max(abs(updated-t)))
        t=updated
        if change<tolerance_cm:break
    b,d=bspline_basis_many(t,knots,derivatives=(0,1))
    f=b@coefficients;m=d@coefficients
    tangent=((x-t)+(y-f)*m)/np.sqrt(1+m*m)
    return (t,tangent,b,d) if return_basis else (t,tangent)


def fit_joint_lane(evidence,domain_cm,initial_center_cm,nominal_width_cm,config=None):
    """Fit a small cubic graph spline and a shared normal-width model.

    ``evidence`` dictionaries contain side, Nx2 points_cm, partial and confidence.
    Per-dataset weights are normalized by support and occupied forward bins, not
    paint thickness. Bounded damped Gauss-Newton alternates with normal matching.
    Diagnostics retain actual row parameters, residuals and accepted weights.
    """
    cfg=config or {};lo,hi=map(float,domain_cm);span=hi-lo
    n=int(cfg.get('joint_control_points',7));nominal=float(nominal_width_cm)
    if not np.isfinite([lo,hi,nominal]).all() or hi<=lo or nominal<=0:
        raise ValueError('Finite increasing domain and positive width required')
    knots=clamped_knots((lo,hi),n)
    init=np.asarray(initial_center_cm,dtype=float)
    if init.ndim!=2 or init.shape[1]!=2 or len(init)<3 or not np.isfinite(init).all():
        raise ValueError('Finite initial current-center samples required')
    init=init[np.argsort(init[:,0])]
    grid=np.linspace(lo,hi,max(64,n*8));bgrid=bspline_basis(grid,knots)
    init_y=np.interp(grid,init[:,0],init[:,1])
    start=np.linalg.lstsq(bgrid,init_y,rcond=None)[0]
    model=JointLaneModel(start.copy(),knots,(lo,hi),nominal,0.)
    if span<float(cfg.get('min_path_span_cm',20.)):
        model.reason='joint_domain_too_short';return model

    datasets=[];point_blocks=[];sides=[];base_blocks=[];ranges=[]
    max_rows=int(cfg.get('joint_max_rows_per_evidence',240))
    partial_factor=float(cfg.get('joint_partial_weight',.2))
    for j,item in enumerate(evidence):
        if item.get('inferred',False):continue
        side=item.get('side');p=np.asarray(item.get('points_cm',[]),dtype=float)
        confidence=float(item.get('confidence',1.));partial=bool(item.get('partial',False))
        if side not in ('left','right') or p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():
            raise ValueError('Finite Nx2 actual boundary rows and a valid side required')
        if not np.isfinite(confidence) or confidence<0:raise ValueError('Nonnegative finite confidence required')
        if len(p)<6 or confidence==0:continue
        order=np.argsort(p[:,0]);p=p[order]
        # Deterministic row subsampling keeps fitting cost bounded; records still
        # identify exactly which actual rows were selected, never generated rows.
        selected=np.arange(len(p))
        if len(p)>max_rows:selected=np.unique(np.linspace(0,len(p)-1,max_rows).round().astype(int));p=p[selected]
        bins=np.clip(((p[:,0]-lo)/span*12).astype(int),0,11)
        counts=np.bincount(bins,minlength=12)
        density=1/np.maximum(counts[bins],1)
        support=min(span,float(np.ptp(p[:,0])))
        strength=100.*min(1.,support/span)*min(confidence,1.)*(partial_factor if partial else 1.)
        if partial:strength=min(strength,float(cfg.get('joint_partial_max_strength',8.)))
        base=density*strength/max(float(density.sum()),1e-12)
        offset=sum(len(q) for q in point_blocks)
        ranges.append((offset,offset+len(p)))
        datasets.append({'source_index':j,'side':side,'partial':partial,'points':p,
            'selected_sorted_rows':selected,'strength':strength})
        point_blocks.append(p);sides.append(np.full(len(p),1. if side=='left' else -1.));base_blocks.append(base)
    if not datasets or not any(not d['partial'] for d in datasets):
        model.reason='no_current_full_boundary_evidence';return model
    points=np.vstack(point_blocks);sign=np.concatenate(sides);base=np.concatenate(base_blocks)
    full_sides={d['side'] for d in datasets if not d['partial']}
    all_sides={d['side'] for d in datasets}
    width_prior=float(cfg.get('joint_width_prior',.15))
    if len(full_sides)==1:
        width_prior=float(cfg.get('joint_partial_width_prior',1.)) if len(all_sides)>1 else float(cfg.get('joint_single_width_prior',10.))
    width_gradient_prior=float(cfg.get('joint_width_slope_prior',3.))
    curvature_reg=float(cfg.get('joint_curvature_regularization',.002))
    jerk_reg=float(cfg.get('joint_jerk_regularization',.0002))
    initial_reg=float(cfg.get('joint_initial_regularization',.0001))
    damping=float(cfg.get('joint_damping',.01));huber=float(cfg.get('joint_huber_cm',2.))
    cap=float(cfg.get('joint_width_max_adjustment_cm',15.));gradient_cap=float(cfg.get('joint_width_max_gradient',.12))
    affine=bool(cfg.get('joint_width_affine',True)) and len(all_sides)>1
    halfspan=span/2;mid=(lo+hi)/2
    if min(width_prior,width_gradient_prior,curvature_reg,jerk_reg,initial_reg,damping,huber,cap,gradient_cap)<=0:
        raise ValueError('Positive joint fit priors, regularizers and limits required')
    theta=np.r_[start,0.,0.]
    b2=bspline_basis(grid,knots,derivative=2)*span**2
    b3=bspline_basis(grid,knots,derivative=3)*span**3
    fixed=np.r_[np.full(n,.001),.001,.001]
    best_loss=float('inf');iterations=0
    projection_iterations=int(cfg.get('joint_projection_iterations',8))
    projection_tolerance=float(cfg.get('joint_projection_tolerance_cm',1e-5))
    outer_iterations=int(cfg.get('joint_iterations',7))
    if projection_iterations<1 or outer_iterations<1 or not np.isfinite(projection_tolerance) or projection_tolerance<=0:
        raise ValueError('Positive joint iteration limits and finite projection tolerance required')

    def bound(values):
        v=values.copy()
        if not affine:v[n+1]=0.
        peak=max(abs(v[n]-v[n+1]),abs(v[n]+v[n+1]))
        factor=min(1.,cap/max(peak,1e-12),gradient_cap*halfspan/max(abs(v[n+1]),1e-12))
        v[n:]*=factor
        return v

    def residual(values,b0,b1,z,weights):
        c=values[:n];f=b0@c;m=b1@c;q=np.sqrt(1+m*m)
        width=nominal+values[n]+values[n+1]*z
        distance=(-(points[:,0]-parameters)*m+(points[:,1]-f))/q
        # Normal projection eliminates arbitrary along-curve correspondence.
        # Fitting both XY coordinates with fixed t would falsely penalize a
        # tangent change that the next closest-point update can absorb.
        data=(distance-.5*sign*width)*np.sqrt(weights)
        return np.r_[data,np.sqrt(curvature_reg/len(grid))*(b2@c),
            np.sqrt(jerk_reg/len(grid))*(b3@c),np.sqrt(initial_reg)*(c-start),
            np.sqrt(width_prior)*values[n],np.sqrt(width_gradient_prior)*values[n+1]]

    projection_initial=None
    for iteration in range(outer_iterations):
        parameters,tangent,b0,b1=_project_center(points,theta[:n],knots,(lo,hi),
            initial=projection_initial,max_iterations=projection_iterations,tolerance_cm=projection_tolerance,return_basis=True)
        projection_initial=parameters
        z=(parameters-mid)/halfspan
        f=b0@theta[:n];m=b1@theta[:n];q=np.sqrt(1+m*m)
        width=nominal+theta[n]+theta[n+1]*z
        predicted=np.column_stack((parameters-.5*sign*width*m/q,f+.5*sign*width/q))
        error=np.linalg.norm(predicted-points,axis=1)
        weights=base*np.minimum(1.,huber/np.maximum(error,1e-12))
        # Endpoint clamping is not a normal correspondence and cannot pull a
        # whole segment toward an observation outside the current valid domain.
        weights[abs(tangent)>float(cfg.get('joint_endpoint_tangent_limit_cm',3.))]=0.
        if np.count_nonzero(weights)>0:
            current=residual(theta,b0,b1,z,weights)
        else:break
        loss=float(current@current)
        jac=np.column_stack([(residual(bound(theta+np.eye(len(theta))[k]*fixed[k]),b0,b1,z,weights)-current)/fixed[k]
            for k in range(len(theta))])
        system=jac.T@jac+damping*np.eye(len(theta));rhs=-jac.T@current
        try:step=np.linalg.solve(system,rhs)
        except np.linalg.LinAlgError:break
        maximum=float(np.max(abs(step[:n])));step*=min(1.,10./max(maximum,1e-12))
        improved=False
        for scale in (1.,.5,.25,.1):
            candidate=bound(theta+scale*step)
            trial=residual(candidate,b0,b1,z,weights);trial_loss=float(trial@trial)
            if np.isfinite(trial_loss) and trial_loss<=loss+1e-9:
                theta=candidate;best_loss=trial_loss;improved=True;break
        iterations=iteration+1
        if not improved or np.max(abs(scale*step))<1e-4:break

    model.coefficients=theta[:n].copy();model.width_mid_cm=float(nominal+theta[n])
    model.width_slope_cm_per_cm=float(theta[n+1]/halfspan)
    parameters,tangent,final_b0,final_b1=_project_center(points,theta[:n],knots,(lo,hi),
        initial=projection_initial,max_iterations=projection_iterations,tolerance_cm=projection_tolerance,return_basis=True)
    left=model.boundary('left',parameters);right=model.boundary('right',parameters)
    predicted=np.where((sign>0)[:,None],left,right)
    error=np.linalg.norm(predicted-points,axis=1)
    robust=base*np.minimum(1.,huber/np.maximum(error,1e-12))
    robust[abs(tangent)>float(cfg.get('joint_endpoint_tangent_limit_cm',3.))]=0.
    total=float(robust.sum());rmse=float(np.sqrt(np.sum(robust*error*error)/max(total,1e-12)))
    final_residual=residual(theta,final_b0,final_b1,(parameters-mid)/halfspan,robust)
    final_objective=float(final_residual@final_residual)
    inlier_limit=float(cfg.get('joint_inlier_limit_cm',4.))
    evidence_info=[];full_support_ok=True
    for d,(a,b) in zip(datasets,ranges):
        supported=(robust[a:b]>0)&(error[a:b]<=float(cfg.get('joint_max_row_error_cm',15.)))
        if not d['partial'] and np.count_nonzero(supported)<max(6,int(.4*(b-a))):full_support_ok=False
        info={'source_index':d['source_index'],'side':d['side'],'partial':d['partial'],
            'selected_sorted_rows':d['selected_sorted_rows'].tolist(),
            'projected_parameter_cm':parameters[a:b].tolist(),'weights':robust[a:b].tolist(),
            'residual_cm':error[a:b].tolist(),'inlier':((error[a:b]<=inlier_limit)&(robust[a:b]>0)).tolist(),
            'used_rows':int(np.count_nonzero(supported)),
            'rmse_cm':float(np.sqrt(np.mean(error[a:b]**2))),
            'support_span_cm':float(np.ptp(parameters[a:b][supported])) if supported.any() else 0.}
        evidence_info.append(info)
    t=np.linspace(lo,hi,257);slope=model.slope(t);q=np.sqrt(1+slope*slope)
    k=model.curvature(t);width=model.width(t);width_prime=model.width_slope_cm_per_cm
    # Exact x-derivatives of C +/- half-width*N (N_x=-f'/q).
    offset_dx=.5*width_prime*slope/q+.5*width*k
    min_forward=float(np.min(np.r_[1-offset_dx,1+offset_dx]))
    finite=bool(np.isfinite(model.center(t)).all() and np.isfinite(k).all() and np.isfinite(width).all())
    model.diagnostics={'iterations':iterations,'weighted_rmse_cm':rmse,
        'objective':final_objective if np.isfinite(final_objective) else None,
        'last_fixed_correspondence_objective':best_loss if np.isfinite(best_loss) else None,
        'projection_iterations_limit':projection_iterations,'projection_tolerance_cm':projection_tolerance,
        'actual_selected_rows':len(points),'full_sides':sorted(full_sides),
        'partial_selected_rows':sum(len(d['points']) for d in datasets if d['partial']),
        'width_min_cm':float(width.min()),'width_max_cm':float(width.max()),
        'width_gradient':model.width_slope_cm_per_cm,'max_abs_center_curvature_per_cm':float(np.max(abs(k))),
        'min_boundary_forward_derivative':min_forward,'full_support_ok':full_support_ok,
        'evidence':evidence_info,'generated_observation_rows':0}
    model.evidence_weights=evidence_info
    if not finite:model.reason='nonfinite_joint_geometry'
    elif not full_support_ok:model.reason='insufficient_current_full_boundary_agreement'
    elif total<=0:model.reason='no_current_normal_correspondences'
    elif rmse>float(cfg.get('joint_max_fit_rmse_cm',8.)):model.reason='joint_residual_too_large'
    elif min_forward<=float(cfg.get('offset_singularity_margin',.05)):model.reason='joint_boundary_forward_fold'
    elif float(np.max(abs(k)))>float(cfg.get('joint_max_abs_curvature_per_cm',.02)):model.reason='joint_curvature_guard'
    else:model.accepted=True;model.reason='joint_current_real_rows_bspline_and_normal_width'
    return model
