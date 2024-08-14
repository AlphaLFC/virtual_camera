import cv2
import numpy as np

from functools import partial
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation


def get_extrinsic_from_euler(x, y, z, pitch, yaw, roll):
    R = Rotation.from_euler(
        'xyz', (pitch, yaw, roll), degrees=True
    ).as_matrix()
    t = np.float32([x, y, z])
    return R, t


def get_extrinsic_from_quant(x, y, z, qx, qy, qz, qw):
    R = Rotation.from_quant((qx, qy, qz, qw)).as_matrix()
    t = np.float32([x, y, z])
    return R, t


def get_intrinsic_mat(cx, cy, fx, fy):
    K = np.float32([
        [fx, 0, cx], 
        [0, fy, cy], 
        [0,  0,  1]
    ])
    return K


def proj_func(x, params):
    p0, p1, p2, p3 = params
    return x + p0 * x**3 + p1 * x**5 + p2 * x**7 + p3 * x**9


def poly_odd6(x, k0, k1, k2, k3, k4, k5):
    return x + k0 * x**3 + k1 * x**5 + k2 * x**7 + k3 * x**9 + k4 * x**11 + k5 * x**13


def get_unproj_func(p0, p1, p2, p3, fov=200):
    theta = np.linspace(-0.5 * fov * np.pi / 180,  0.5 * fov * np.pi / 180, 2000)
    theta_d = proj_func(theta, (p0, p1, p2, p3))
    params, pcov = curve_fit(poly_odd6, theta_d, theta)
    error = np.sqrt(np.diag(pcov)).mean()
    assert error < 1e-2, "poly parameter curve fitting failed: {:f}.".format(error)
    k0, k1, k2, k3, k4, k5 = params
    return partial(poly_odd6, k0=k0, k1=k1, k2=k2, k3=k3, k4=k4, k5=k5)


def ext_motovis2image(ext_motovis):
    x, y, z, pitch, yaw, roll = ext_motovis
    return x, -z, y, 90 + pitch, - roll, yaw


class BaseCamera:
    """
    Camera Coordinate System: image-style, normalized coords.
        - motovis-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    """
    def __init__(self, resolution, extrinsic, intrinsic, ego_mask=None):
        """## Args:
        - resolution : tuple (w, h)
        - extrinsic : list or tuple (R, t) in motovis-style ego system
        - intrinsic : list or tuple (cx, cy, fx, fy, <distortion params>)
        - ego_mask : in shape (h, w)
        """
        self.resolution = resolution
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self._init_ext_int_mat()
        self.ego_mask = ego_mask
        self.camera_mask = None
    
    def _init_ext_int_mat(self):
        self.R_e, self.t_e = self.extrinsic
        self.T_e = np.eye(4)
        self.T_e[:3, :3] = self.R_e
        self.T_e[:3, 3] = self.t_e
        
        cx, cy, fx, fy = self.intrinsic[:4]
        self.K = np.float32([
            [fx, 0, cx], 
            [0, fy, cy], 
            [0,  0,  1]
        ])
    
    def project_points_from_camera_to_image(self, camera_points):
        raise NotImplementedError

    def unproject_points_from_image_to_camera(self):
        raise NotImplementedError
    
    def get_camera_mask(self):
        """
        Returns a mask of the camera's view.
        """
        if self.camera_mask is None:
            self.camera_mask = self.ego_mask
        return self.camera_mask



class FisheyeCamera(BaseCamera):
    """
    Camera Coordinate System: image-style, normalized coords.
        - motovis-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    """
    def __init__(self, resolution, extrinsic, intrinsic, fov=None, ego_mask=None):
        """## Args:
        - resolution : tuple (w, h)
        - extrinsic : list or tuple (R, t) in motovis-style ego system
        - intrinsic : list or tuple (cx, cy, fx, fy, p0, p1, p2, p3)
        - fov : float, in degree
        - ego_mask : in shape (h, w)
        """
        super().__init__(resolution, extrinsic, intrinsic, ego_mask=ego_mask)
        if fov is None:
            self.fov = 225
        else:
            self.fov = fov

    def project_points_from_camera_to_image(self, camera_points):
        # camera_points in image-style: x-y-z right-down-forward
        cx, cy, fx, fy, p0, p1, p2, p3 = self.intrinsic
        xx = camera_points[0]
        yy = camera_points[1]
        zz = camera_points[2]
        # distance to camera center ray
        dd = np.sqrt(xx**2 + yy**2)
        # radius(focal=1) to light center point, aka theta between ray and center ray
        rr = theta = np.arctan2(dd, zz)
        # rr = theta = np.clip(np.arctan2(dd, zz), -self.fov / 2 * np.pi / 180, self.fov / 2 * np.pi / 180)
        fov_mask = np.logical_and(theta >= -self.fov / 2 * np.pi / 180, theta <= self.fov / 2 * np.pi / 180)
    
        # projected coords on fisheye camera image
        r_distorted = theta_distorted = proj_func(theta, (p0, p1, p2, p3))
        uu = np.float32(fx * (r_distorted * xx / dd) + cx)
        vv = np.float32(fy * (r_distorted * yy / dd) + cy)
        uu[~fov_mask] = -1
        vv[~fov_mask] = -1
        return uu, vv


    def unproject_points_from_image_to_camera(self):
        W, H = self.resolution
        cx, cy, fx, fy, p0, p1, p2, p3 = self.intrinsic
        unproj_func = get_unproj_func(p0, p1, p2, p3, fov=self.fov)
        
        uu, vv = np.meshgrid(
            np.linspace(0, W - 1, W), 
            np.linspace(0, H - 1, H)
        )
        x_distorted = (uu - cx) / fx
        y_distorted = (vv - cy) / fy
        
        # r_distorted = theta_distorted
        r_distorted = np.sqrt(x_distorted**2 + y_distorted**2)
        # r_distorted[r_distorted < 1e-5] = 1e-5
        theta = unproj_func(r_distorted)
        # theta = np.clip(theta, - 0.5 * self.fov * np.pi / 180, 0.5 * self.fov * np.pi / 180)
        self.camera_mask = np.float32(np.abs(theta * 180 / np.pi) < self.fov / 2)
    
        # get camera coords by ray intersecting with a sphere in image-style (x-y-z right-down-forward)
        r_distorted[r_distorted < 1e-5] = 1e-5
        dd = np.sin(theta)
        xx = x_distorted * dd / r_distorted
        yy = y_distorted * dd / r_distorted
        zz = np.cos(theta)
        
        camera_points = np.stack([xx, yy, zz], axis=0).reshape(3, -1)

        return camera_points
    

    def get_camera_mask(self, use_fov_mask=False):
        """
        Returns a mask of the camera's view.
        """
        if self.camera_mask is None and use_fov_mask:
            W, H = self.resolution
            cx, cy, fx, fy, p0, p1, p2, p3 = self.intrinsic
            unproj_func = get_unproj_func(p0, p1, p2, p3, fov=self.fov)
            
            uu, vv = np.meshgrid(
                np.linspace(0, W - 1, W), 
                np.linspace(0, H - 1, H)
            )
            x_distorted = (uu - cx) / fx
            y_distorted = (vv - cy) / fy
            
            # r_distorted = theta_distorted
            r_distorted = np.sqrt(x_distorted**2 + y_distorted**2)
            r_distorted[r_distorted < 1e-5] = 1e-5
            theta = unproj_func(r_distorted)
            self.camera_mask = np.float32(np.abs(theta * 180 / np.pi) < self.fov / 2)
        
            if self.ego_mask is not None:
                self.camera_mask *= self.ego_mask
        else:
            self.camera_mask = self.ego_mask
    
        return self.camera_mask

    
    def _to_motovis_cfg(self):
        cfg_camera = {}
        cfg_camera['sensor_model'] = 'src.sensors.cameras.OpenCVFisheyeCamera'
        cfg_camera['image_size'] = self.resolution
        quant = Rotation.from_matrix(self.R_e).as_quat()
        cfg_camera['extrinsic'] = list(self.t_e) + list(quant)
        cfg_camera['pp'] = self.intrinsic[:2]
        cfg_camera['focal'] = self.intrinsic[2:4]
        cfg_camera['inv_poly'] = self.intrinsic[4:]
        cfg_camera['fov_fit'] = self.fov
        return cfg_camera


    @classmethod
    def init_from_motovis_cfg(cls, cfg_camera, use_default_fov=True):
        camera_model = cfg_camera['sensor_model']
        assert camera_model in ['src.sensors.cameras.OpenCVFisheyeCamera']

        resolution = cfg_camera['image_size']
        # ego system is in MOTOVIS-style
        t_e = cfg_camera['extrinsic'][:3]
        R_e = Rotation.from_quat(cfg_camera['extrinsic'][3:]).as_matrix()
        extrinsic = (R_e, t_e)

        #cx, cy, fx, fy, p0, p1, p2, p3
        intrinsic = cfg_camera['pp'] + cfg_camera['focal'] + cfg_camera['inv_poly']
        if use_default_fov:
            fov = None
        else:
            fov = cfg_camera['fov_fit']

        return cls(resolution, extrinsic, intrinsic, fov)
    



class PerspectiveCamera(BaseCamera):
    """
    Camera Coordinate System: image-style, normalized coords.
        - motovis-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    """
    def __init__(self, resolution, extrinsic, intrinsic, ego_mask=None):
        """## Args:
        - resolution : tuple (w, h)
        - extrinsic : list or tuple (R, t) in motovis-style ego system
        - intrinsic : list or tuple (cx, cy, fx, fy)
        - ego_mask : in shape (h, w)
        """
        super().__init__(resolution, extrinsic, intrinsic, ego_mask=ego_mask)
    
    def unproject_points_from_image_to_camera(self):
        W, H = self.resolution
        cx, cy, fx, fy = self.intrinsic
        
        uu, vv = np.meshgrid(
            np.linspace(0, W - 1, W), 
            np.linspace(0, H - 1, H)
        )
        # get camera coords by ray intersecting with a z-plane in image-style (x-y-z right-down-forward)
        xx = (uu - cx) / fx
        yy = (vv - cy) / fy
        zz = np.ones_like(uu)

        camera_points = np.stack([xx, yy, zz], axis=0).reshape(3, -1)

        return camera_points

    def project_points_from_camera_to_image(self, camera_points):
        img_points = np.matmul(self.K, camera_points.reshape(3, -1)).reshape(camera_points.shape)
        img_points[2, np.abs(img_points[2]) < 1e-5] = 1e-5
        uu = np.float32(img_points[0] / img_points[2])
        vv = np.float32(img_points[1] / img_points[2])
        return uu, vv
    
    def _to_motovis_cfg(self):
        cfg_camera = {}
        cfg_camera['sensor_model'] = 'src.sensors.cameras.PerspectiveCamera'
        cfg_camera['image_size'] = self.resolution
        quant = Rotation.from_matrix(self.R_e).as_quat()
        cfg_camera['extrinsic'] = list(self.t_e) + list(quant)
        cfg_camera['pp'] = self.intrinsic[:2]
        cfg_camera['focal'] = self.intrinsic[2:4]
        return cfg_camera

    @classmethod
    def init_from_nuscense_cfg(cls, cfg):
        pass

    @classmethod
    def init_from_av2_cfg(cls, cfg):
        pass

    @classmethod
    def init_from_motovis_cfg(cls, cfg_camera):
        camera_model = cfg_camera['sensor_model']
        assert camera_model in [
            'src.sensors.cameras.PerspectiveCamera', 
            'src.sensors.cameras.PinholeCamera',
            'src.sensors.cameras.DDADPerspectiveCamera',
            'src.sensors.cameras.NuScenesPerspectiveCamera'
        ]

        resolution = cfg_camera['image_size']
        # ego system is in MOTOVIS-style
        t_e = cfg_camera['extrinsic'][:3]
        R_e = Rotation.from_quat(cfg_camera['extrinsic'][3:]).as_matrix()
        extrinsic = (R_e, t_e)

        #cx, cy, fx, fy
        intrinsic = cfg_camera['pp'] + cfg_camera['focal']

        return cls(resolution, extrinsic, intrinsic)



AVAILABLE_CAMERA_TYPES = [FisheyeCamera, PerspectiveCamera]



def _check_camera_type(camera):
    return any([isinstance(camera, camera_type) for camera_type in AVAILABLE_CAMERA_TYPES])


def render_image(src_img, src_camera, dst_camera, interpolation=cv2.INTER_LINEAR):
    assert _check_camera_type(src_camera), 'AssertError: src_camera must be one of {}'.format(AVAILABLE_CAMERA_TYPES)
    assert _check_camera_type(dst_camera), 'AssertError: dst_camera must be one of {}'.format(AVAILABLE_CAMERA_TYPES)

    # assert src_img.shape[:2][::-1] == src_camera.resolution, 'AssertError: src_image must have the same resolution as src_camera'

    R_e_src = src_camera.R_e
    R_e_dst = dst_camera.R_e

    R_dst_src = R_e_dst.T @ R_e_src

    dst_camera_points = dst_camera.unproject_points_from_image_to_camera()

    rot_dst_camera_points = R_dst_src.T @ dst_camera_points

    uu, vv = src_camera.project_points_from_camera_to_image(rot_dst_camera_points)
    src_camera_mask = src_camera.get_camera_mask()

    dst_img = cv2.remap(
        src_img, 
        uu.reshape(dst_camera.resolution[::-1]),
        vv.reshape(dst_camera.resolution[::-1]),
        interpolation=interpolation
    )
    
    src_img_mask = np.ones(src_img.shape[:2], dtype=np.float32)
    if src_camera_mask is not None:
        src_img_mask *= src_camera_mask
    dst_img_mask = cv2.remap(
        src_img_mask, 
        uu.reshape(dst_camera.resolution[::-1]),
        vv.reshape(dst_camera.resolution[::-1]),
        interpolation=cv2.INTER_NEAREST
    )

    return dst_img, dst_img_mask


def create_virtual_perspective_camera(resolution, euler_angles, translations, intrinsic='auto'):
    W, H = resolution
    if intrinsic == 'auto':
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 2
        intrinsic = (cx, cy, fx, fy)
    # ego system, in motovis-style, x-y-z right-forward-up
    R = Rotation.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    t = translations
    return PerspectiveCamera(resolution, (R, t), intrinsic)


def create_virtual_fisheye_camera(resolution, euler_angles, translations, intrinsic='auto'):
    # inv_poly: [0.05345955558134785, -0.005850248788053312, -0.0005388425917994607, -0.0001609567223788042]
    W, H = resolution
    if intrinsic == 'auto':
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 4
        intrinsic = (cx, cy, fx, fy, 0.1, 0, 0, 0)
    # ego system, in motovis-style, x-y-z right-forward-up
    R = Rotation.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    t = translations
    return FisheyeCamera(resolution, (R, t), intrinsic)
        

VCAMERA_PERSPECTIVE_FRONT = create_virtual_perspective_camera((1280, 960), (-90, 0, 0), (0, 1.5, 1.5))
VCAMERA_PERSPECTIVE_FRONT_LEFT = create_virtual_perspective_camera((1280, 960), (-90, 0, 45), (-1, 2, 1))
VCAMERA_PERSPECTIVE_FRONT_RIGHT = create_virtual_perspective_camera((1280, 960), (-90, 0, -45), (1, 2, 1))
VCAMERA_PERSPECTIVE_BACK = create_virtual_perspective_camera((1280, 960), (-90, 0, 180), (0, -1, 1))
VCAMERA_PERSPECTIVE_BACK_LEFT = create_virtual_perspective_camera((1280, 960), (-90, 0, 135), (-1, 2, 1))
VCAMERA_PERSPECTIVE_BACK_RIGHT = create_virtual_perspective_camera((1280, 960), (-90, 0, -135), (1, 2, 1))

VCAMERA_FISHEYE_FRONT = create_virtual_fisheye_camera((1024, 640), (-120, 0, 0), (0, 3.5, 0.5))
VCAMERA_FISHEYE_LEFT = create_virtual_fisheye_camera((1024, 640), (-135, 0, 90), (-1, 2, 1))
VCAMERA_FISHEYE_RIGHT = create_virtual_fisheye_camera((1024, 640), (-135, 0, -90), (1, 2, 1))
VCAMERA_FISHEYE_BACK = create_virtual_fisheye_camera((1024, 640), (-120, 0, 180), (0, -1, 0.5))