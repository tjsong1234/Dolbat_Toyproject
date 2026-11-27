"""Rear-axle reference anchored to the ORIGINAL image bottom, not BEV bottom."""
import numpy as np


def rear_reference(homography, mapping, image_size, rear_to_image_bottom_m=.9):
    """Ground-plane translation with explicit centreline and heading assumptions.

    image_size is (width,height) of the frame used by the homography. We assume
    its bottom midpoint is on the vehicle centreline and the BEV forward axis
    is parallel to vehicle heading. Neither alignment was measured in the PDF.
    """
    width,height=image_size
    h=np.asarray(homography,dtype=float)
    distance=float(rear_to_image_bottom_m)
    if width<2 or height<2 or h.shape!=(3,3) or not np.isfinite(h).all() or not np.isfinite(distance) or distance<=0:
        raise ValueError('Finite homography, positive distance and valid image size required')
    pixel=np.array([(width-1)/2,height-1,1.])
    transformed=h@pixel
    if abs(transformed[2])<1e-12:raise ValueError('Original image bottom maps to infinity')
    uv=transformed[:2]/transformed[2]
    cm=mapping.pixels_to_cm(uv[None,:])[0]
    translation=np.array([distance,0.])-cm/100
    return {
        'source_bottom_center_px':pixel[:2].tolist(),
        'source_bottom_bev_px':uv.tolist(),
        'source_bottom_bev_cm':cm.tolist(),
        'rear_translation_m':translation.tolist(),
        'rear_to_image_bottom_m':distance,
        'source_pdf':'20261126_돌밭자율주행_입문스터디.pdf',
        'source_pdf_pages':[126,127],
        'assumptions':[
            'Original image bottom midpoint lies on the vehicle centreline.',
            'Vehicle heading is parallel to the calibrated BEV forward axis.',
            f'Original image bottom is {distance:g} m forward of the rear axle, on the road plane.',
            'The planar BEV metric scale remains applicable at that image anchor.',
        ],
        'note':f'{distance:g} m is the configured course distance; centreline/heading alignment is assumed, not measured.'
    }
