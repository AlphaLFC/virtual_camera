import numpy as np

from virtual_camera import PerspectiveCamera, render_image


def test_render_image_preserves_input_rank():
    camera = PerspectiveCamera(
        resolution=(4, 3),
        extrinsic=(np.eye(3), np.zeros(3)),
        intrinsic=(2.0, 1.5, 1.0, 1.0),
    )
    rendered_2d, mask_2d = render_image(np.zeros((3, 4), dtype=np.uint8), camera, camera)
    rendered_3d, mask_3d = render_image(np.zeros((3, 4, 1), dtype=np.uint8), camera, camera)

    assert rendered_2d.shape == (3, 4)
    assert rendered_3d.shape == (3, 4, 1)
    assert mask_2d.shape == mask_3d.shape == (3, 4)
