"""Real row summaries from six formerly kinked video frames; no fabricated pixels."""
import json
from pathlib import Path
import numpy as np
import pytest
from dolbat_core.lane_detection import LaneFit,LaneResult
from dolbat_core.metric_mapping import MetricMapping
from dolbat_core.yellow_guided_planning import YellowGuidedLanePlanner
from test_yellow_guided_planning import CONFIG

FIXTURE=json.loads((Path(__file__).parent/'fixtures/week3_bend_observations.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case',FIXTURE['cases'],ids=lambda c:str(c['frame']))
def test_actual_problem_frames_keep_full_model_without_local_curvature_spike(case):
    mapping=MetricMapping(**FIXTURE['mapping'])
    def lane(side):
        points=mapping.cm_to_pixels(np.asarray(case['observed_row_centers_cm'][side]))
        co=np.polyfit(points[:,1],points[:,0],2)
        return LaneFit(co,points,0,float(np.ptp(points[:,1])/639),float(points[:,1].min()),float(points[:,1].max()))
    lanes=LaneResult(lane('left'),lane('right'),'DETECTED','saved real row-centre observations',1,
        np.zeros((640,800,3),np.uint8),np.zeros(800))
    p=YellowGuidedLanePlanner(mapping,CONFIG,90).plan(lanes)
    assert p.usable and p.yellow_guided and len(p.full_model_parameter_cm)==CONFIG['samples']
    np.testing.assert_allclose(p.full_model_parameter_cm[[0,-1]],p.right.interval_cm,atol=1e-8)
    # Empirical regression guard for these recorded scenes, not a vehicle limit.
    assert max(abs(p.raw_curvature_per_cm))<.01
    assert max(abs(p.curvature_per_cm))<.01
    assert p.boundary_pair_midpoint_error_cm<1e-10
