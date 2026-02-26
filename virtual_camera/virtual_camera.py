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


def ftheta_forward_poly(theta, coeffs):
    """ray2pixel"""
    r = 0
    for i, c in enumerate(coeffs):
        r += c * (theta ** i)
    return r


def ftheta_backward_poly(r, coeffs):
    """pixel2ray"""
    theta = 0
    for i, c in enumerate(coeffs):
        theta += c * (r ** i)
    return theta



def ext_motovis2image(ext_motovis):
    x, y, z, pitch, yaw, roll = ext_motovis
    return x, -z, y, 90 + pitch, - roll, yaw


class BaseCamera:
    """
    Camera Coordinate System: image-style, normalized coords.
        - lidar-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    """
    def __init__(self, resolution, extrinsic, intrinsic, ego_mask=None):
        """## Args:
        - resolution : tuple (w, h)
        - extrinsic : list or tuple (R, t)
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
    
    def __repr__(self):
        return f'{self.__class__.__name__}(resolution={self.resolution}, extrinsic={self.T_e}, intrinsic={self.intrinsic})'



class FisheyeCamera(BaseCamera):
    """
    Camera Coordinate System: image-style, normalized coords.
        - lidar-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    """
    def __init__(self, resolution, extrinsic, intrinsic, fov=None, ego_mask=None):
        """## Args:
        - resolution : tuple (w, h)
        - extrinsic : list or tuple (R, t)
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
        # ego system is in lidar-style
        t_e = cfg_camera['extrinsic'][:3]
        R_e = Rotation.from_quat(cfg_camera['extrinsic'][3:]).as_matrix()
        extrinsic = (R_e, t_e)

        #cx, cy, fx, fy, p0, p1, p2, p3
        intrinsic = cfg_camera['pp'] + cfg_camera['focal'] + cfg_camera['inv_poly'][:4]
        if use_default_fov:
            fov = None
        else:
            fov = cfg_camera['fov_fit']

        return cls(resolution, extrinsic, intrinsic, fov)
    


class FThetaCamera(BaseCamera):
    """
    Kannala-Brandt F-Theta camera model for fisheye cameras.
    
    Uses the Kannala-Brandt generic fisheye model with odd-power polynomials:
        Forward (ray->pixel):  r(θ) = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
        Backward (pixel->ray): θ(r) = r + j1*r³ + j2*r⁵ + j3*r⁷ + j4*r⁹
    
    Where r is the normalized image radius (pixel distance / focal_length),
    θ is the angle between the incoming ray and the optical axis,
    and fx, fy are separate focal length parameters for pixel scaling.
    
    Pixel projection:
        u = fx * (r(θ) / ρ) * X + cx
        v = fy * (r(θ) / ρ) * Y + cy
    where ρ = sqrt(X² + Y²)
    
    Camera Coordinate System: image-style, normalized coords.
        - lidar-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    
    Intrinsic format: (cx, cy, fx, fy, sign, *coeffs)
        - 'F': Forward only, coeffs = [k1, k2, k3, k4]  (4 distortion coefficients)
        - 'B': Backward only, coeffs = [j1, j2, j3, j4]
        - 'FB': Both, coeffs = [k1, k2, k3, k4, j1, j2, j3, j4]
    
    Note: The linear term (coefficient=1) is implicit in the polynomial.
    The coefficients k1..k4 / j1..j4 are for powers 3, 5, 7, 9 respectively.
    """
    
    def __init__(self, resolution, extrinsic, intrinsic, fov=None, ego_mask=None):
        """
        Args:
            resolution: tuple (w, h)
            extrinsic: tuple (R, t)
            intrinsic: tuple (cx, cy, fx, fy, sign, *coeffs)
                - sign: 'F', 'B', or 'FB'
                - coeffs: distortion coefficients (4 for F/B, 8 for FB)
            fov: float or None - field of view in degrees (auto-computed if None)
            ego_mask: ndarray in shape (h, w)
        """
        super().__init__(resolution, extrinsic, intrinsic, ego_mask=ego_mask)
        
        # Parse intrinsic parameters
        self.cx, self.cy, self.fx, self.fy = intrinsic[:4]
        self.poly_sign = intrinsic[4]
        coeffs = list(intrinsic[5:])
        
        # Validate and parse polynomial coefficients
        self.forward_coeffs = None
        self.backward_coeffs = None
        
        if self.poly_sign == 'F':
            # Forward only: 4 distortion coefficients for θ³, θ⁵, θ⁷, θ⁹
            assert len(coeffs) >= 4, f"Forward only requires 4 coefficients, got {len(coeffs)}"
            self.forward_coeffs = coeffs[:4]
            # Auto-fit backward polynomial
            self.backward_coeffs = self._fit_backward_from_forward()
            
        elif self.poly_sign == 'B':
            # Backward only: 4 distortion coefficients for r³, r⁵, r⁷, r⁹
            assert len(coeffs) >= 4, f"Backward only requires 4 coefficients, got {len(coeffs)}"
            self.backward_coeffs = coeffs[:4]
            # Auto-fit forward polynomial
            self.forward_coeffs = self._fit_forward_from_backward()
            
        elif self.poly_sign == 'FB':
            # Both forward and backward: 8 coefficients total
            assert len(coeffs) >= 8, f"Both requires 8 coefficients, got {len(coeffs)}"
            self.forward_coeffs = coeffs[:4]
            self.backward_coeffs = coeffs[4:8]
            # Validate polynomial consistency
            self._validate_polynomial_pair()
        else:
            raise ValueError(f"Invalid polynomial sign '{self.poly_sign}'. Must be 'F', 'B', or 'FB'")
        
        # Set or compute FOV
        if fov is None:
            self.fov = self._compute_fov()
        else:
            self.fov = fov
    
    @staticmethod
    def _evaluate_forward_poly(x, coeffs):
        """
        Evaluate Kannala-Brandt odd-power polynomial:
            f(x) = x + k1*x³ + k2*x⁵ + k3*x⁷ + k4*x⁹
        
        The linear term (coefficient=1) is implicit.
        coeffs = [k1, k2, k3, k4] for powers [3, 5, 7, 9].
        """
        x = np.asarray(x, dtype=np.float64)
        result = x.copy()  # Linear term: 1*x
        x2 = x * x
        x_power = x2 * x  # x³
        for c in coeffs:
            result += c * x_power
            x_power *= x2  # next odd power
        return result
    
    @staticmethod
    def _evaluate_backward_poly(x, coeffs):
        """
        Evaluate Kannala-Brandt backward odd-power polynomial:
            g(x) = x + j1*x³ + j2*x⁵ + j3*x⁷ + j4*x⁹
        
        Same structure as forward — the linear term is implicit.
        coeffs = [j1, j2, j3, j4] for powers [3, 5, 7, 9].
        """
        x = np.asarray(x, dtype=np.float64)
        result = x.copy()  # Linear term: 1*x
        x2 = x * x
        x_power = x2 * x  # x³
        for c in coeffs:
            result += c * x_power
            x_power *= x2  # next odd power
        return result
    
    def _fit_backward_from_forward(self, num_samples=2000):
        """Fit backward polynomial from forward polynomial using curve fitting.
        
        Given forward: r(θ) = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
        Fit backward:  θ(r) = r + j1*r³ + j2*r⁵ + j3*r⁷ + j4*r⁹
        """
        # Sample theta values over valid range
        theta_max = np.pi  # Up to 180 degrees
        theta_samples = np.linspace(1e-6, theta_max, num_samples)
        
        # Compute corresponding r values using forward polynomial
        r_samples = self._evaluate_forward_poly(theta_samples, self.forward_coeffs)
        
        # Only fit where r is monotonically increasing and positive
        dr = np.diff(r_samples)
        valid_end = np.searchsorted(dr <= 0, True)
        if valid_end < 10:
            return [0.0, 0.0, 0.0, 0.0]
        
        r_fit = r_samples[:valid_end]
        theta_fit = theta_samples[:valid_end]
        
        # Fit: θ(r) = r + j1*r³ + j2*r⁵ + j3*r⁷ + j4*r⁹
        # Residual = θ - r, fit with odd powers of r
        def backward_residual(r, j1, j2, j3, j4):
            r2 = r * r
            return r + j1*r2*r + j2*r2*r2*r + j3*r2*r2*r2*r + j4*r2*r2*r2*r2*r
        
        try:
            popt, _ = curve_fit(backward_residual, r_fit, theta_fit, p0=[0, 0, 0, 0])
            return list(popt)
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]
    
    def _fit_forward_from_backward(self, num_samples=2000):
        """Fit forward polynomial from backward polynomial using curve fitting.
        
        Given backward: θ(r) = r + j1*r³ + j2*r⁵ + j3*r⁷ + j4*r⁹
        Fit forward:    r(θ) = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
        """
        # Estimate maximum normalized radius from image
        W, H = self.resolution
        max_r = np.sqrt((W / 2 / self.fx) ** 2 + (H / 2 / self.fy) ** 2) * 1.2
        
        # Sample r values
        r_samples = np.linspace(1e-6, max_r, num_samples)
        
        # Compute corresponding theta values using backward polynomial
        theta_samples = self._evaluate_backward_poly(r_samples, self.backward_coeffs)
        
        # Only fit where theta is monotonically increasing and positive
        dtheta = np.diff(theta_samples)
        valid_end = np.searchsorted(dtheta <= 0, True)
        if valid_end < 10:
            return [0.0, 0.0, 0.0, 0.0]
        
        theta_fit = theta_samples[:valid_end]
        r_fit = r_samples[:valid_end]
        
        # Fit: r(θ) = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
        def forward_residual(theta, k1, k2, k3, k4):
            t2 = theta * theta
            return theta + k1*t2*theta + k2*t2*t2*theta + k3*t2*t2*t2*theta + k4*t2*t2*t2*t2*theta
        
        try:
            popt, _ = curve_fit(forward_residual, theta_fit, r_fit, p0=[0, 0, 0, 0])
            return list(popt)
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]
    
    def _validate_polynomial_pair(self, tolerance=1e-3):
        """Validate that forward and backward polynomials are approximate inverses."""
        theta_test = np.linspace(0.01, np.pi / 2, 100)
        
        # Forward then backward: θ -> r -> θ_recovered
        r_forward = self._evaluate_forward_poly(theta_test, self.forward_coeffs)
        theta_recovered = self._evaluate_backward_poly(r_forward, self.backward_coeffs)
        
        max_error = np.max(np.abs(theta_recovered - theta_test))
        if max_error > tolerance:
            print(f"Warning: Polynomial pair validation failed. Max round-trip error: {max_error:.6f} rad")
    
    def _compute_fov(self):
        """Compute maximum valid FOV from forward polynomial monotonicity."""
        theta_test = np.linspace(0, np.pi, 1000)
        r_test = self._evaluate_forward_poly(theta_test, self.forward_coeffs)
        
        # Find where dr/dθ becomes non-positive
        dr = np.diff(r_test)
        valid_indices = np.where(dr > 1e-6)[0]
        
        if len(valid_indices) > 0:
            max_theta = theta_test[valid_indices[-1]]
            return min(max_theta * 2 * 180 / np.pi, 270)  # Full FOV = 2 * max_theta
        else:
            return 200  # Default FOV
    
    def project_points_from_camera_to_image(self, camera_points):
        """
        Forward projection: 3D camera coordinates -> 2D image coordinates.
        
        Uses Kannala-Brandt model:
            θ = atan2(ρ, Z)   where ρ = sqrt(X² + Y²)
            r(θ) = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
            u = fx * (r(θ)/ρ) * X + cx
            v = fy * (r(θ)/ρ) * Y + cy
        
        Args:
            camera_points: ndarray of shape (3, N) in image-style coords
                (x-y-z right-down-forward)
        
        Returns:
            uu, vv: image coordinates
        """
        xx = camera_points[0]
        yy = camera_points[1]
        zz = camera_points[2]
        
        # Distance to optical axis
        dd = np.sqrt(xx**2 + yy**2)
        
        # Angle between ray and optical axis
        theta = np.arctan2(dd, zz)
        
        # Apply FOV mask
        fov_mask = np.logical_and(
            theta >= -self.fov / 2 * np.pi / 180,
            theta <= self.fov / 2 * np.pi / 180
        )
        
        # Compute normalized radius using forward polynomial: r(θ)
        r_distorted = self._evaluate_forward_poly(theta, self.forward_coeffs)
        
        # Avoid division by zero
        dd_safe = np.where(dd < 1e-6, 1e-6, dd)
        
        # Project to image coordinates
        uu = np.float32(self.fx * (r_distorted * xx / dd_safe) + self.cx)
        vv = np.float32(self.fy * (r_distorted * yy / dd_safe) + self.cy)
        
        # Mask out-of-FOV points
        uu[~fov_mask] = -1
        vv[~fov_mask] = -1
        
        return uu, vv
    
    def unproject_points_from_image_to_camera(self):
        """
        Backward projection: 2D image coordinates -> 3D camera rays.
        
        Uses Kannala-Brandt model:
            x' = (u - cx) / fx,  y' = (v - cy) / fy
            r_d = sqrt(x'² + y'²)
            θ = r_d + j1*r_d³ + j2*r_d⁵ + j3*r_d⁷ + j4*r_d⁹
            ray = [x'/r_d * sin(θ), y'/r_d * sin(θ), cos(θ)]
        
        Returns:
            camera_points: ndarray of shape (3, H*W) in image-style coords
        """
        W, H = self.resolution
        
        # Create image coordinate grid
        uu, vv = np.meshgrid(
            np.linspace(0, W - 1, W),
            np.linspace(0, H - 1, H)
        )
        
        # Normalize to camera coordinates
        x_distorted = (uu - self.cx) / self.fx
        y_distorted = (vv - self.cy) / self.fy
        
        # Compute normalized radius from principal point
        r_distorted = np.sqrt(x_distorted**2 + y_distorted**2)
        
        # Compute theta using backward polynomial: θ(r)
        theta = self._evaluate_backward_poly(r_distorted, self.backward_coeffs)
        
        # Update camera mask
        self.camera_mask = np.float32(np.abs(theta * 180 / np.pi) < self.fov / 2)
        
        # Avoid division by zero
        r_safe = np.where(r_distorted < 1e-6, 1e-6, r_distorted)
        
        # Compute 3D ray direction
        sin_theta = np.sin(theta)
        xx = x_distorted * sin_theta / r_safe
        yy = y_distorted * sin_theta / r_safe
        zz = np.cos(theta)
        
        camera_points = np.stack([xx, yy, zz], axis=0).reshape(3, -1)
        return camera_points
    
    def get_camera_mask(self, use_fov_mask=False):
        """
        Returns a mask of the camera's view.
        
        Args:
            use_fov_mask: if True, compute mask based on FOV
        
        Returns:
            mask: ndarray of shape (H, W)
        """
        if self.camera_mask is None and use_fov_mask:
            _ = self.unproject_points_from_image_to_camera()
            
            if self.ego_mask is not None:
                self.camera_mask *= self.ego_mask
        else:
            self.camera_mask = self.ego_mask
        
        return self.camera_mask
    
    def _to_motovis_cfg(self):
        """Export to motovis-compatible config dict (as OpenCVFisheyeCamera)."""
        cfg_camera = {}
        cfg_camera['sensor_model'] = 'src.sensors.cameras.OpenCVFisheyeCamera'
        cfg_camera['image_size'] = list(self.resolution)
        quant = Rotation.from_matrix(self.R_e).as_quat()
        cfg_camera['extrinsic'] = list(self.t_e) + list(quant)
        cfg_camera['pp'] = [self.cx, self.cy]
        cfg_camera['focal'] = [self.fx, self.fy]
        cfg_camera['inv_poly'] = list(self.forward_coeffs)
        cfg_camera['fov_fit'] = self.fov
        return cfg_camera
    
    @classmethod
    def init_from_nvidia_cfg(cls, cfg, extrinsic=None, fov=None, ego_mask=None):
        """
        Initialize from NVIDIA PhysicalAI dataset format.
        
        NVIDIA format stores fx, fy as separate fields (or derives from polynomial).
        Polynomial coefficients are the Kannala-Brandt distortion terms.
        
        Args:
            cfg: dict with keys from NVIDIA parquet format:
                - width, height: image dimensions
                - cx, cy: principal point
                - fx, fy: focal lengths (optional, derived from polynomial if absent)
                - bw_poly_0..bw_poly_3: backward polynomial distortion coefficients
                - fw_poly_0..fw_poly_3: forward polynomial distortion coefficients (optional)
            extrinsic: tuple (R, t) or None
            fov: float or None
            ego_mask: ndarray or None
        
        Returns:
            FThetaCamera instance
        """
        resolution = (cfg['width'], cfg['height'])
        cx, cy = cfg['cx'], cfg['cy']
        
        # Get focal lengths: explicit > derived > default
        if 'fx' in cfg and 'fy' in cfg:
            fx, fy = cfg['fx'], cfg['fy']
        elif 'fx' in cfg:
            fx = fy = cfg['fx']
        else:
            # Derive from backward polynomial: for small r, θ ≈ r, so fx ≈ 1/j0_linear
            # But in Kannala-Brandt the linear term is 1, so fx must come from elsewhere.
            # Use a reasonable default based on image size.
            fx = fy = max(cfg['width'], cfg['height']) / 2.0
        
        # Extract polynomial coefficients (4 distortion terms for odd powers)
        has_forward = all(f'fw_poly_{i}' in cfg for i in range(4))
        has_backward = all(f'bw_poly_{i}' in cfg for i in range(4))
        
        if has_forward and has_backward:
            forward_coeffs = [cfg[f'fw_poly_{i}'] for i in range(4)]
            backward_coeffs = [cfg[f'bw_poly_{i}'] for i in range(4)]
            intrinsic = (cx, cy, fx, fy, 'FB') + tuple(forward_coeffs) + tuple(backward_coeffs)
        elif has_forward:
            forward_coeffs = [cfg[f'fw_poly_{i}'] for i in range(4)]
            intrinsic = (cx, cy, fx, fy, 'F') + tuple(forward_coeffs)
        elif has_backward:
            backward_coeffs = [cfg[f'bw_poly_{i}'] for i in range(4)]
            intrinsic = (cx, cy, fx, fy, 'B') + tuple(backward_coeffs)
        else:
            raise ValueError("NVIDIA cfg must contain either fw_poly or bw_poly coefficients")
        
        if extrinsic is None:
            R = np.eye(3)
            t = np.zeros(3)
            extrinsic = (R, t)
        
        return cls(resolution, extrinsic, intrinsic, fov=fov, ego_mask=ego_mask)
    
    @classmethod
    def init_from_motovis_cfg(cls, cfg_camera, use_default_fov=True):
        """
        Initialize from motovis calibration config.
        
        The motovis format stores the Kannala-Brandt forward distortion coefficients
        in the 'inv_poly' field (confusingly named in the config).
        
        Args:
            cfg_camera: dict with keys: sensor_model, image_size, extrinsic,
                        focal, pp, inv_poly, fov_fit
            use_default_fov: if True, auto-compute FOV; else use fov_fit from config
        
        Returns:
            FThetaCamera instance
        """
        camera_model = cfg_camera['sensor_model']
        assert camera_model in ['src.sensors.cameras.OpenCVFisheyeCamera']
        
        resolution = cfg_camera['image_size']
        t_e = cfg_camera['extrinsic'][:3]
        R_e = Rotation.from_quat(cfg_camera['extrinsic'][3:]).as_matrix()
        extrinsic = (R_e, t_e)
        
        cx, cy = cfg_camera['pp']
        fx, fy = cfg_camera['focal']
        inv_poly = cfg_camera['inv_poly']
        
        # inv_poly are the Kannala-Brandt forward distortion coefficients (k1, k2, k3, k4)
        # Pad to 4 coefficients if shorter
        forward_coeffs = list(inv_poly)
        while len(forward_coeffs) < 4:
            forward_coeffs.append(0.0)
        forward_coeffs = forward_coeffs[:4]
        
        intrinsic = (cx, cy, fx, fy, 'F') + tuple(forward_coeffs)
        
        if use_default_fov:
            fov = None
        else:
            fov = cfg_camera.get('fov_fit', None)
        
        return cls(resolution, extrinsic, intrinsic, fov=fov)


class PerspectiveCamera(BaseCamera):
    """
    Camera Coordinate System: image-style, normalized coords.
        - lidar-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    """
    def __init__(self, resolution, extrinsic, intrinsic, ego_mask=None):
        """## Args:
        - resolution : tuple (w, h)
        - extrinsic : list or tuple (R, t)
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
        mask = (uu >= 0) * (uu < self.resolution[0]) * (vv >= 0) * (vv < self.resolution[1]) * (camera_points[2] > 1e-3)
        uu[~mask] = -1
        vv[~mask] = -1
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
        # ego system is in lidar-style
        t_e = cfg_camera['extrinsic'][:3]
        R_e = Rotation.from_quat(cfg_camera['extrinsic'][3:]).as_matrix()
        extrinsic = (R_e, t_e)

        #cx, cy, fx, fy
        intrinsic = cfg_camera['pp'] + cfg_camera['focal']

        return cls(resolution, extrinsic, intrinsic)



class BrownConradyCamera(BaseCamera):
    """
    Camera Coordinate System: image-style, normalized coords.
        - lidar-style: x-y-z right-forward-up
        - openGL-style: x-y-z right-up-backward
        - image-style: x-y-z right-down-forward
        - pytorch3d-style: x-y-z left-up-forward
    """
    def __init__(self, resolution, extrinsic, intrinsic):
        """## Args:
        - resolution : tuple (w, h)
        - extrinsic : list or tuple (R, t)
        - intrinsic : list or tuple (cx, cy, fx, fy, k1, k2, p1, p2, <k3, ...>)
        """
        super().__init__(resolution, extrinsic, intrinsic)


    def unproject_points_from_image_to_camera(self):
        W, H = self.resolution
        dist_coeffs = np.float32(self.intrinsic[4:])

        uu, vv = np.meshgrid(
            np.linspace(0, W - 1, W), 
            np.linspace(0, H - 1, H)
        )

        distorted_points = np.stack([uu, vv], axis=-1).reshape(-1, 1, 2)

        undistorted_points = cv2.undistortPoints(
            src=distorted_points,
            cameraMatrix=self.K,
            distCoeffs=dist_coeffs,
            P=None
        ).reshape(-1, 2)

        camera_points = np.stack([
            undistorted_points[:, 0], 
            undistorted_points[:, 1], 
            np.ones_like(uu)
        ], axis=0).reshape(3, -1)
        
        return camera_points

    
    def project_points_from_camera_to_image(self, camera_points):
        dist_coeffs = np.float32(self.intrinsic[4:])
        
        xx = camera_points[0]
        yy = camera_points[1]
        zz = camera_points[2]

        valid_zz = zz > 1e-3

        uu = np.full_like(xx, -1, dtype=np.float32)
        vv = np.full_like(yy, -1, dtype=np.float32)

        if np.sum(valid_zz) == 0:
            return uu, vv

        points_3d = np.stack([xx[valid_zz], yy[valid_zz], zz[valid_zz]], axis=-1).reshape(-1, 1, 3)

        image_points, _ = cv2.projectPoints(
            points_3d, 
            np.zeros((3,1), dtype=np.float32), 
            np.zeros((3,1), dtype=np.float32), 
            self.K, 
            dist_coeffs
        )

        uu[valid_zz] = image_points[:, 0, 0]
        vv[valid_zz] = image_points[:, 0, 1]

        valid_mask = (uu >= 0) * (uu < self.resolution[0]) * (vv >= 0) * (vv < self.resolution[1])
        uu[~valid_mask] = -1
        vv[~valid_mask] = -1
        
        return uu, vv


PinholeCamera = BrownConradyCamera


AVAILABLE_CAMERA_TYPES = [FisheyeCamera, FThetaCamera, PerspectiveCamera, BrownConradyCamera, PinholeCamera]



def _check_camera_type(camera):
    return any([isinstance(camera, camera_type) for camera_type in AVAILABLE_CAMERA_TYPES])


def render_image(src_img, src_camera, dst_camera, interpolation=cv2.INTER_LINEAR, dump_uu_vv=False):
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
    if dump_uu_vv:
        return dst_img, dst_img_mask, uu, vv
    return dst_img, dst_img_mask


def convert_fisheye_to_ftheta(fisheye_camera):
    """
    Convert a FisheyeCamera to a FThetaCamera.
    
    Both models use the same Kannala-Brandt projection:
        r(θ) = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
    
    The FisheyeCamera stores intrinsic as (cx, cy, fx, fy, p0, p1, p2, p3)
    where p0..p3 are the distortion coefficients for θ³, θ⁵, θ⁷, θ⁹.
    
    Args:
        fisheye_camera: FisheyeCamera instance
    
    Returns:
        FThetaCamera instance with equivalent projection
    """
    assert isinstance(fisheye_camera, FisheyeCamera), \
        f"Expected FisheyeCamera, got {type(fisheye_camera).__name__}"
    
    cx, cy, fx, fy = fisheye_camera.intrinsic[:4]
    distortion_coeffs = list(fisheye_camera.intrinsic[4:])
    
    # Pad to 4 coefficients
    while len(distortion_coeffs) < 4:
        distortion_coeffs.append(0.0)
    distortion_coeffs = distortion_coeffs[:4]
    
    # Build FThetaCamera intrinsic: (cx, cy, fx, fy, 'F', k1, k2, k3, k4)
    intrinsic = (cx, cy, fx, fy, 'F') + tuple(distortion_coeffs)
    
    return FThetaCamera(
        resolution=fisheye_camera.resolution,
        extrinsic=(fisheye_camera.R_e, fisheye_camera.t_e),
        intrinsic=intrinsic,
        fov=fisheye_camera.fov,
        ego_mask=fisheye_camera.ego_mask
    )


def convert_ftheta_to_fisheye(ftheta_camera):
    """
    Convert a FThetaCamera to a FisheyeCamera.
    
    Both models use the same Kannala-Brandt projection:
        r(θ) = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
    
    The FThetaCamera stores forward distortion coefficients as [k1, k2, k3, k4].
    The FisheyeCamera uses (cx, cy, fx, fy, p0, p1, p2, p3).
    
    If the FThetaCamera was constructed with only backward coefficients,
    this function uses the auto-fitted forward coefficients.
    
    Args:
        ftheta_camera: FThetaCamera instance
    
    Returns:
        FisheyeCamera instance with equivalent projection
    """
    assert isinstance(ftheta_camera, FThetaCamera), \
        f"Expected FThetaCamera, got {type(ftheta_camera).__name__}"
    
    # Use forward coefficients (always available, auto-fitted if needed)
    forward_coeffs = list(ftheta_camera.forward_coeffs)
    while len(forward_coeffs) < 4:
        forward_coeffs.append(0.0)
    forward_coeffs = forward_coeffs[:4]
    
    # Build FisheyeCamera intrinsic: (cx, cy, fx, fy, p0, p1, p2, p3)
    intrinsic = (ftheta_camera.cx, ftheta_camera.cy,
                 ftheta_camera.fx, ftheta_camera.fy) + tuple(forward_coeffs)
    
    return FisheyeCamera(
        resolution=ftheta_camera.resolution,
        extrinsic=(ftheta_camera.R_e, ftheta_camera.t_e),
        intrinsic=intrinsic,
        fov=ftheta_camera.fov,
        ego_mask=ftheta_camera.ego_mask
    )


def refit_ftheta_from_fisheye(fisheye_camera, num_samples=2000):
    """
    Read a FisheyeCamera model and refit its distortion as an FThetaCamera
    with both forward and backward polynomials.
    
    This creates an FThetaCamera where:
    - Forward coefficients come directly from the FisheyeCamera distortion
    - Backward coefficients are fitted to be the inverse
    
    Args:
        fisheye_camera: FisheyeCamera instance
        num_samples: number of sample points for polynomial fitting
    
    Returns:
        FThetaCamera instance with both forward and backward polynomials
    """
    assert isinstance(fisheye_camera, FisheyeCamera), \
        f"Expected FisheyeCamera, got {type(fisheye_camera).__name__}"
    
    ftheta = convert_fisheye_to_ftheta(fisheye_camera)
    
    # The FThetaCamera constructor already auto-fits backward from forward
    # Verify round-trip quality
    theta_test = np.linspace(0.01, ftheta.fov / 2 * np.pi / 180, 100)
    r_forward = FThetaCamera._evaluate_forward_poly(theta_test, ftheta.forward_coeffs)
    theta_recovered = FThetaCamera._evaluate_backward_poly(r_forward, ftheta.backward_coeffs)
    max_error = np.max(np.abs(theta_recovered - theta_test))
    
    if max_error > 0.01:  # > 0.01 radians ≈ 0.57 degrees
        print(f"Warning: refit round-trip error is {max_error:.6f} rad ({np.degrees(max_error):.4f}°)")
    
    return ftheta


def refit_fisheye_from_ftheta(ftheta_camera, num_samples=2000):
    """
    Read an FThetaCamera model and refit its distortion as a FisheyeCamera.
    
    If the FThetaCamera was created from backward-only coefficients,
    this uses the auto-fitted forward coefficients.
    
    Args:
        ftheta_camera: FThetaCamera instance
        num_samples: number of sample points for polynomial fitting
    
    Returns:
        FisheyeCamera instance with forward polynomial (distortion) coefficients
    """
    assert isinstance(ftheta_camera, FThetaCamera), \
        f"Expected FThetaCamera, got {type(ftheta_camera).__name__}"
    
    fisheye = convert_ftheta_to_fisheye(ftheta_camera)
    
    # Verify projection consistency by comparing projections at test angles
    theta_test = np.linspace(0.01, ftheta_camera.fov / 2 * np.pi / 180, 100)
    
    # FThetaCamera forward projection
    r_ftheta = FThetaCamera._evaluate_forward_poly(theta_test, ftheta_camera.forward_coeffs)
    
    # FisheyeCamera forward projection (same formula)
    r_fisheye = proj_func(theta_test, tuple(fisheye.intrinsic[4:8]))
    
    max_error = np.max(np.abs(r_ftheta - r_fisheye))
    if max_error > 1e-6:
        print(f"Warning: projection mismatch between ftheta and fisheye: {max_error:.8f}")
    
    return fisheye

def create_virtual_perspective_camera(resolution, euler_angles, translations, intrinsic='auto'):
    W, H = resolution
    if intrinsic == 'auto':
        cx = (W - 1) / 2
        cy = (H - 1) / 2
        fx = fy = W / 2
        intrinsic = (cx, cy, fx, fy)
    # ego system, in lidar-style, x-y-z right-forward-up
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
    # ego system, in lidar-style, x-y-z right-forward-up
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