from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import yaml
from dolbat_core.metric_mapping import MetricMapping
from dolbat_core.metric_planning import MetricLanePlanner
from dolbat_core.boundary_pair_planning import attach_boundary_pair
from dolbat_core.lane_detection import LaneFit,LaneResult
from dolbat_core.final_visualization import green_overlay,metric_with_partials,boundary_curves,metric_view_bounds,boundary_visibility

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'dolbat_core/resources/config/week3_final.yaml').read_text(encoding='utf-8'))['planning']


def single_plan():
    v=np.repeat(np.arange(80,601),7);u=580+np.tile(np.arange(-3,4),521)
    right=LaneFit(np.array([0.,0.,580]),np.column_stack((u,v)),0,.8,80,600)
    result=LaneResult(None,right,'INFERRED_LEFT','test',1,np.zeros((640,800,3),np.uint8),np.zeros(800))
    mapping=MetricMapping(.25,.5,399.5,639)
    plan=MetricLanePlanner(mapping,CFG,90).plan(result)
    return attach_boundary_pair(plan,mapping,CFG,90),mapping


def test_estimated_blue_boundary_does_not_make_green_center_dashed():
    plan,mapping=single_plan();im=green_overlay(np.zeros((640,800,3),np.uint8),plan,mapping)
    points=mapping.cm_to_pixels(plan.path_cm);x=round(points[:,0].mean())
    lo=int(np.ceil(points[:,1].min()))+3;hi=int(np.floor(points[:,1].max()))-3
    # Every row is green, including rows between the old dashed path strokes.
    patch=im[lo:hi,x-1:x+2];green=(patch[:,:,1]>150)&(patch[:,:,1]>patch[:,:,0]*1.5)&(patch[:,:,1]>patch[:,:,2]*1.5)
    assert np.all(green.any(axis=1))
    curves=boundary_curves(plan)
    assert next(inferred for side,_,_,inferred in curves if side=='left')
    assert plan.left is None and plan.boundary_pair_left_estimated


def test_live_metric_center_is_continuous_for_one_observed_boundary():
    plan,_=single_plan();im=metric_with_partials(plan,2)
    b=metric_view_bounds(plan);scale=min((480-75)/(2*b['y_limit']),(360-65)/(b['x_max']-b['x_min']))
    x=round(240-plan.path_cm[:,1].mean()*scale)
    top=25+((360-65)-(b['x_max']-b['x_min'])*scale)/2
    ys=top+(b['x_max']-plan.path_cm[:,0])*scale;lo=int(np.ceil(ys.min()))+3;hi=int(np.floor(ys.max()))-3
    p=im[lo:hi,x-1:x+2];green=(p[:,:,1]>150)&(p[:,:,1]>p[:,:,0]*1.5)&(p[:,:,1]>p[:,:,2]*1.5)
    assert green.any(axis=1).all()


def test_missing_current_observation_does_not_draw_old_green_center():
    plan,mapping=single_plan();plan.path_cm=np.empty((0,2));plan.left=plan.right=None
    plan.inferred_boundary_cm=None;plan.boundary_pair_left_cm=plan.boundary_pair_right_cm=np.empty((0,2))
    image=green_overlay(np.zeros((640,800,3),np.uint8),plan,mapping)
    assert not np.any(image)


def test_full_blue_model_has_priority_over_short_observed_interval():
    plan,mapping=single_plan()
    full=plan.boundary_pair_left_cm.copy();plan.full_model_left_cm=full
    plan.full_model_right_cm=plan.boundary_pair_right_cm.copy()
    plan.full_model_left_inferred_mask=np.ones(len(full),bool)
    from dolbat_core.metric_planning import MetricBoundary
    plan.left=MetricBoundary(np.array([0.,0.,45.]),np.empty((0,2)),np.empty((0,2)),np.array([],bool),(100,130),0,0)
    blue=next(points for side,points,_,_ in boundary_curves(plan) if side=='left')
    np.testing.assert_array_equal(blue,full)
    assert np.ptp(blue[:,0])>200


def test_offscreen_boundary_is_reported_without_moving_its_geometry():
    points=np.array([[-50.,20.],[-30.,350.]])
    result=boundary_visibility((480,640,3),points)
    assert not result['visible'] and result['side']=='left'
    crossing=boundary_visibility((480,640,3),np.array([[-20.,250.],[660.,250.]]))
    assert crossing['visible'] and crossing['visible_pixels']>600
