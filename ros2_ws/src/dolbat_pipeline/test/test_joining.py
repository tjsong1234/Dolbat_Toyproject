from dolbat_pipeline.joining import FrameJoiner
import pytest


def test_out_of_order_streams_are_joined_in_frame_order():
    j=FrameJoiner(('a','b'),4)
    assert j.add('a',1,(0,1),'a1')==[]
    assert j.add('b',1,(0,1),'b1')==[]
    assert j.add('a',0,(0,0),'a0')==[]
    assert j.add('b',0,(0,0),'b0')==[(0,{'a':'a0','b':'b0'}),(1,{'a':'a1','b':'b1'})]
    j.finish(2)


def test_mismatched_headers_and_missing_stream_are_errors():
    j=FrameJoiner(('a','b'),2);j.add('a',0,(0,1),'a')
    with pytest.raises(ValueError,match='timestamp'):j.add('b',0,(0,2),'b')
    with pytest.raises(ValueError,match='cache'):j.add('a',2,(0,3),'a2')
    with pytest.raises(ValueError,match='Incomplete'):j.finish(1)


def test_completed_frame_redelivery_is_idempotent_and_cache_stays_empty():
    j=FrameJoiner(('a',),2)
    for i in range(100):assert j.add('a',i,(0,i),i)==[(i,{'a':i})]
    assert j.add('a',0,(0,0),0)==[]
    assert not j.pending and not j.stamps
    j.finish(100)
