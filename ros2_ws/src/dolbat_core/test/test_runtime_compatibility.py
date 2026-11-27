import ast
from pathlib import Path
import numpy as np
from dolbat_core import metric_mapping
from dolbat_core.runtime import Runtime


def test_numpy_1x_integration_fallback(monkeypatch):
    calls=[]
    monkeypatch.delattr(np,'trapezoid',raising=False)
    def legacy(y,x):
        calls.append((y,x));return 12.5
    monkeypatch.setattr(np,'trapz',legacy,raising=False)
    assert metric_mapping._trapezoid([1,2],[0,1])==12.5
    assert calls==[([1,2],[0,1])]


def test_numpy_2x_uses_current_function_without_legacy_lookup(monkeypatch):
    monkeypatch.setattr(np,'trapezoid',lambda y,x:7.,raising=False)
    monkeypatch.delattr(np,'trapz',raising=False)
    assert metric_mapping._trapezoid([1,2],[0,1])==7.


def test_resources_and_original_bottom_rear_anchor():
    runtime=Runtime()
    reference=np.array(runtime.geometry['source_bottom_bev_cm'])/100+runtime.translation
    np.testing.assert_allclose(reference,[.9,0],atol=1e-12)
    np.testing.assert_allclose(runtime.translation,[.5456472037113449,.06718881926837683],atol=1e-12)
    assert runtime.control_config.lookahead_m==1.7
    assert runtime.control_config.wheelbase_m==.55
    assert runtime.geo['preprocessing']['bev_size']==[800,640]


def test_python_310_syntax_for_all_core_modules():
    module=Path(metric_mapping.__file__).parent
    for path in module.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,10))
