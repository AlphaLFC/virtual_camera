"""
Test suite for FThetaCamera implementation.
Tests round-trip consistency, polynomial validation, conversion methods,
and real-data verification with motovis calibration files.
"""
import numpy as np
import sys
import os
import yaml

sys.path.insert(0, '/home/alpha/Projects/VirtualCamera')

from virtual_camera import (
    FThetaCamera, FisheyeCamera, PerspectiveCamera,
    render_image, convert_fisheye_to_ftheta, convert_ftheta_to_fisheye,
    refit_ftheta_from_fisheye, refit_fisheye_from_ftheta
)
from scipy.spatial.transform import Rotation


def test_forward_only():
    """Test FThetaCamera with forward polynomial only (Kannala-Brandt)."""
    print("\n=== Test 1: Forward Only ===")
    
    # Kannala-Brandt forward distortion: r(θ) = θ + k1*θ³ + k2*θ⁵ + k3*θ⁷ + k4*θ⁹
    # coeffs = [k1, k2, k3, k4] for odd powers 3, 5, 7, 9
    # fx/fy are separate focal lengths for pixel scaling
    intrinsic = (960, 540, 500.0, 500.0, 'F', 
                 0.05, -0.001, 0.0001, 0.0)
    
    R = np.eye(3)
    t = np.zeros(3)
    
    camera = FThetaCamera(
        resolution=(1920, 1080),
        extrinsic=(R, t),
        intrinsic=intrinsic,
        fov=200
    )
    
    print(f"Forward coeffs: {camera.forward_coeffs}")
    print(f"Backward coeffs (fitted): {camera.backward_coeffs}")
    print(f"FOV: {camera.fov} degrees")
    
    test_passed = test_round_trip(camera)
    return test_passed


def test_backward_only():
    """Test FThetaCamera with backward polynomial only."""
    print("\n=== Test 2: Backward Only ===")
    
    # Backward distortion: θ(r) = r + j1*r³ + j2*r⁵ + j3*r⁷ + j4*r⁹
    intrinsic = (960, 540, 500.0, 500.0, 'B',
                 -0.04, 0.0008, -0.00005, 0.0)
    
    R = np.eye(3)
    t = np.zeros(3)
    
    camera = FThetaCamera(
        resolution=(1920, 1080),
        extrinsic=(R, t),
        intrinsic=intrinsic,
        fov=200
    )
    
    print(f"Backward coeffs: {camera.backward_coeffs}")
    print(f"Forward coeffs (fitted): {camera.forward_coeffs}")
    print(f"FOV: {camera.fov} degrees")
    
    test_passed = test_round_trip(camera)
    return test_passed


def test_both_polynomials():
    """Test FThetaCamera with both forward and backward polynomials."""
    print("\n=== Test 3: Both Polynomials ===")
    
    forward_coeffs = [0.05, -0.001, 0.0001, 0.0]
    backward_coeffs = [-0.04, 0.0008, -0.00005, 0.0]
    
    intrinsic = (960, 540, 500.0, 500.0, 'FB') + tuple(forward_coeffs) + tuple(backward_coeffs)
    
    R = np.eye(3)
    t = np.zeros(3)
    
    camera = FThetaCamera(
        resolution=(1920, 1080),
        extrinsic=(R, t),
        intrinsic=intrinsic,
        fov=200
    )
    
    print(f"Forward coeffs: {camera.forward_coeffs}")
    print(f"Backward coeffs: {camera.backward_coeffs}")
    print(f"FOV: {camera.fov} degrees")
    
    test_passed = test_round_trip(camera)
    
    # Validate polynomial consistency
    print("\nValidating polynomial consistency...")
    theta_test = np.linspace(0.01, np.pi/3, 100)
    r_forward = FThetaCamera._evaluate_forward_poly(theta_test, camera.forward_coeffs)
    theta_recovered = FThetaCamera._evaluate_backward_poly(r_forward, camera.backward_coeffs)
    max_error = np.max(np.abs(theta_recovered - theta_test))
    print(f"Max round-trip error: {max_error:.6f} radians ({np.degrees(max_error):.4f}°)")
    
    return test_passed


def test_nvidia_format():
    """Test initialization from NVIDIA dataset format."""
    print("\n=== Test 4: NVIDIA Dataset Format ===")
    
    nvidia_cfg = {
        'width': 1920,
        'height': 1080,
        'cx': 960.0,
        'cy': 540.0,
        'fx': 500.0,
        'fy': 500.0,
        'bw_poly_0': -0.04,
        'bw_poly_1': 0.0008,
        'bw_poly_2': -0.00005,
        'bw_poly_3': 0.0,
        'fw_poly_0': 0.05,
        'fw_poly_1': -0.001,
        'fw_poly_2': 0.0001,
        'fw_poly_3': 0.0,
    }
    
    camera = FThetaCamera.init_from_nvidia_cfg(nvidia_cfg)
    
    print(f"Resolution: {camera.resolution}")
    print(f"Principal point: ({camera.cx}, {camera.cy})")
    print(f"Focal lengths: ({camera.fx}, {camera.fy})")
    print(f"Forward coeffs: {camera.forward_coeffs}")
    print(f"Backward coeffs: {camera.backward_coeffs}")
    
    test_passed = test_round_trip(camera)
    return test_passed


def test_round_trip(camera):
    """
    Test round-trip consistency: project ray → pixel → unproject → ray
    """
    print("\nTesting round-trip consistency...")
    
    all_passed = True
    max_error = 0
    
    for angle_deg in [-60, -45, -30, 0, 30, 45, 60]:
        angle_rad = angle_deg * np.pi / 180
        
        # Create test ray at this angle (in camera coordinates)
        test_ray = np.array([
            [np.sin(angle_rad)],
            [0.0],
            [np.cos(angle_rad)]
        ])
        
        # Project to image
        uu, vv = camera.project_points_from_camera_to_image(test_ray)
        
        # Check if projection is valid
        if uu[0] < 0 or vv[0] < 0 or uu[0] >= camera.resolution[0] or vv[0] >= camera.resolution[1]:
            print(f"  Angle {angle_deg}°: Out of FOV")
            continue
        
        W, H = camera.resolution
        u_pixel = int(round(uu[0]))
        v_pixel = int(round(vv[0]))
        u_pixel = max(0, min(u_pixel, W - 1))
        v_pixel = max(0, min(v_pixel, H - 1))
        
        # Get all camera rays
        camera_rays = camera.unproject_points_from_image_to_camera()
        
        # Extract the ray at this specific pixel
        pixel_idx = v_pixel * W + u_pixel
        recovered_ray = camera_rays[:, pixel_idx]
        
        # Normalize both rays
        test_ray_norm = test_ray[:, 0] / np.linalg.norm(test_ray[:, 0])
        recovered_ray_norm = recovered_ray / np.linalg.norm(recovered_ray)
        
        # Compute angular error
        cos_angle = np.dot(test_ray_norm, recovered_ray_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angular_error = np.arccos(cos_angle) * 180 / np.pi
        
        max_error = max(max_error, angular_error)
        
        if angular_error < 1.0:
            print(f"  Angle {angle_deg}°: ✓ Error = {angular_error:.4f}°")
        else:
            print(f"  Angle {angle_deg}°: ✗ Error = {angular_error:.4f}° (too large)")
            all_passed = False
    
    print(f"Max angular error: {max_error:.4f}°")
    return all_passed and max_error < 1.0


def test_fov_computation():
    """Test automatic FOV computation."""
    print("\n=== Test 5: FOV Computation ===")
    
    intrinsic = (960, 540, 500.0, 500.0, 'F',
                 0.05, -0.001, 0.0001, 0.0)
    
    R = np.eye(3)
    t = np.zeros(3)
    
    camera = FThetaCamera(
        resolution=(1920, 1080),
        extrinsic=(R, t),
        intrinsic=intrinsic
    )
    
    print(f"Auto-computed FOV: {camera.fov:.2f}°")
    print(f"✓ FOV computation works" if camera.fov > 0 else "✗ FOV computation failed")
    
    return camera.fov > 0


def test_polynomial_math():
    """Test the Kannala-Brandt polynomial evaluation directly."""
    print("\n=== Test 6: Polynomial Math ===")
    
    # Test with known coefficients: r(θ) = θ + 0.1*θ³ - 0.01*θ⁵
    coeffs = [0.1, -0.01, 0.0, 0.0]
    
    # At θ=0, r should be 0
    r0 = FThetaCamera._evaluate_forward_poly(np.array([0.0]), coeffs)
    assert abs(r0[0]) < 1e-10, f"r(0) should be 0, got {r0[0]}"
    print(f"  r(0) = {r0[0]:.10f} ✓")
    
    # At θ=1, r should be 1 + 0.1*1 - 0.01*1 = 1.09
    r1 = FThetaCamera._evaluate_forward_poly(np.array([1.0]), coeffs)
    expected = 1.0 + 0.1 * 1.0**3 - 0.01 * 1.0**5
    assert abs(r1[0] - expected) < 1e-10, f"r(1) should be {expected}, got {r1[0]}"
    print(f"  r(1) = {r1[0]:.10f}, expected {expected:.10f} ✓")
    
    # At θ=0.5, r should be 0.5 + 0.1*0.125 - 0.01*0.03125 = 0.51218...
    theta = 0.5
    r_half = FThetaCamera._evaluate_forward_poly(np.array([theta]), coeffs)
    expected = theta + 0.1 * theta**3 - 0.01 * theta**5
    assert abs(r_half[0] - expected) < 1e-10, f"r(0.5) should be {expected}, got {r_half[0]}"
    print(f"  r(0.5) = {r_half[0]:.10f}, expected {expected:.10f} ✓")
    
    # Verify odd powers only (even derivative check)
    # For pure linear (no distortion), r = θ exactly
    zero_coeffs = [0.0, 0.0, 0.0, 0.0]
    theta_arr = np.linspace(0, 2, 100)
    r_linear = FThetaCamera._evaluate_forward_poly(theta_arr, zero_coeffs)
    assert np.allclose(r_linear, theta_arr), "With zero distortion, r should equal θ"
    print(f"  Zero distortion: r = θ ✓")
    
    return True


def test_fisheye_to_ftheta_conversion():
    """Test converting FisheyeCamera to FThetaCamera."""
    print("\n=== Test 7: Fisheye → FTheta Conversion ===")
    
    # Create a FisheyeCamera with known coefficients
    # FisheyeCamera intrinsic: (cx, cy, fx, fy, p0, p1, p2, p3)
    intrinsic = (960, 540, 468.46, 468.54, 
                 0.058, -0.011, 0.0019, -0.0006)
    R = np.eye(3)
    t = np.zeros(3)
    
    fisheye = FisheyeCamera(
        resolution=(1920, 1080),
        extrinsic=(R, t),
        intrinsic=intrinsic,
        fov=190
    )
    
    # Convert to FTheta
    ftheta = convert_fisheye_to_ftheta(fisheye)
    
    print(f"Fisheye distortion: {list(fisheye.intrinsic[4:])}")
    print(f"FTheta forward:     {ftheta.forward_coeffs}")
    print(f"FTheta backward:    {ftheta.backward_coeffs}")
    print(f"FTheta fx={ftheta.fx}, fy={ftheta.fy}")
    
    # Both should produce the same projection for test rays
    test_angles = [10, 30, 45, 60, 80]
    max_pixel_error = 0
    
    for angle_deg in test_angles:
        angle_rad = angle_deg * np.pi / 180
        ray = np.array([[np.sin(angle_rad)], [0.0], [np.cos(angle_rad)]])
        
        uu_fish, vv_fish = fisheye.project_points_from_camera_to_image(ray)
        uu_ft, vv_ft = ftheta.project_points_from_camera_to_image(ray)
        
        pixel_error = np.sqrt((uu_fish[0] - uu_ft[0])**2 + (vv_fish[0] - vv_ft[0])**2)
        max_pixel_error = max(max_pixel_error, pixel_error)
        
        status = "✓" if pixel_error < 0.5 else "✗"
        print(f"  {angle_deg}°: fisheye=({uu_fish[0]:.1f},{vv_fish[0]:.1f}) "
              f"ftheta=({uu_ft[0]:.1f},{vv_ft[0]:.1f}) err={pixel_error:.4f}px {status}")
    
    print(f"Max pixel error: {max_pixel_error:.4f}")
    return max_pixel_error < 1.0


def test_ftheta_to_fisheye_conversion():
    """Test converting FThetaCamera to FisheyeCamera."""
    print("\n=== Test 8: FTheta → Fisheye Conversion ===")
    
    # Create FThetaCamera
    intrinsic = (960, 540, 468.46, 468.54, 'F',
                 0.058, -0.011, 0.0019, -0.0006)
    R = np.eye(3)
    t = np.zeros(3)
    
    ftheta = FThetaCamera(
        resolution=(1920, 1080),
        extrinsic=(R, t),
        intrinsic=intrinsic,
        fov=190
    )
    
    # Convert to FisheyeCamera
    fisheye = convert_ftheta_to_fisheye(ftheta)
    
    print(f"FTheta forward:     {ftheta.forward_coeffs}")
    print(f"Fisheye distortion: {list(fisheye.intrinsic[4:])}")
    
    # Both should produce the same projection
    test_angles = [10, 30, 45, 60, 80]
    max_pixel_error = 0
    
    for angle_deg in test_angles:
        angle_rad = angle_deg * np.pi / 180
        ray = np.array([[np.sin(angle_rad)], [0.0], [np.cos(angle_rad)]])
        
        uu_ft, vv_ft = ftheta.project_points_from_camera_to_image(ray)
        uu_fish, vv_fish = fisheye.project_points_from_camera_to_image(ray)
        
        pixel_error = np.sqrt((uu_ft[0] - uu_fish[0])**2 + (vv_ft[0] - vv_fish[0])**2)
        max_pixel_error = max(max_pixel_error, pixel_error)
        
        status = "✓" if pixel_error < 0.5 else "✗"
        print(f"  {angle_deg}°: ftheta=({uu_ft[0]:.1f},{vv_ft[0]:.1f}) "
              f"fisheye=({uu_fish[0]:.1f},{vv_fish[0]:.1f}) err={pixel_error:.4f}px {status}")
    
    print(f"Max pixel error: {max_pixel_error:.4f}")
    return max_pixel_error < 1.0


def test_refit_roundtrip():
    """Test refit functions preserve projection accuracy."""
    print("\n=== Test 9: Refit Round-Trip ===")
    
    # Start with a FisheyeCamera
    intrinsic = (960, 540, 468.46, 468.54,
                 0.058, -0.011, 0.0019, -0.0006)
    R = np.eye(3)
    t = np.zeros(3)
    
    fisheye_orig = FisheyeCamera(
        resolution=(1920, 1080),
        extrinsic=(R, t),
        intrinsic=intrinsic,
        fov=190
    )
    
    # Refit: fisheye → ftheta → fisheye
    ftheta = refit_ftheta_from_fisheye(fisheye_orig)
    fisheye_recovered = refit_fisheye_from_ftheta(ftheta)
    
    print(f"Original fisheye distortion:  {list(fisheye_orig.intrinsic[4:])}")
    print(f"Intermediate ftheta forward:  {ftheta.forward_coeffs}")
    print(f"Intermediate ftheta backward: {ftheta.backward_coeffs}")
    print(f"Recovered fisheye distortion: {list(fisheye_recovered.intrinsic[4:])}")
    
    # Compare projections
    test_angles = [10, 30, 45, 60, 80]
    max_pixel_error = 0
    
    for angle_deg in test_angles:
        angle_rad = angle_deg * np.pi / 180
        ray = np.array([[np.sin(angle_rad)], [0.0], [np.cos(angle_rad)]])
        
        uu_orig, vv_orig = fisheye_orig.project_points_from_camera_to_image(ray)
        uu_rec, vv_rec = fisheye_recovered.project_points_from_camera_to_image(ray)
        
        pixel_error = np.sqrt((uu_orig[0] - uu_rec[0])**2 + (vv_orig[0] - vv_rec[0])**2)
        max_pixel_error = max(max_pixel_error, pixel_error)
        
        status = "✓" if pixel_error < 0.5 else "✗"
        print(f"  {angle_deg}°: err={pixel_error:.6f}px {status}")
    
    print(f"Max pixel error: {max_pixel_error:.6f}")
    return max_pixel_error < 1.0


def test_motovis_ftheta_init():
    """Test FThetaCamera initialization from motovis calibration config."""
    print("\n=== Test 10: Motovis Config Initialization ===")
    
    # Load real calibration
    calib_path = '/home/alpha/Projects/VirtualCamera/data/motovis/calibration_fix/calibration_ipm_back.yml'
    if not os.path.exists(calib_path):
        print("Calibration file not found, skipping")
        return True
    
    with open(calib_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Test with camera5 (wide-angle fisheye, fov=190)
    camera_cfg = cfg['rig']['camera5']
    
    # Create both FisheyeCamera and FThetaCamera from same config
    fisheye = FisheyeCamera.init_from_motovis_cfg(camera_cfg, use_default_fov=False)
    ftheta = FThetaCamera.init_from_motovis_cfg(camera_cfg, use_default_fov=False)
    
    print(f"Camera: camera5 (fov={camera_cfg['fov_fit']})")
    print(f"Fisheye intrinsic: {fisheye.intrinsic}")
    print(f"FTheta forward: {ftheta.forward_coeffs}")
    print(f"FTheta backward: {ftheta.backward_coeffs}")
    print(f"FTheta fx={ftheta.fx}, fy={ftheta.fy}")
    
    # Compare projections
    test_angles = [10, 30, 45, 60, 80]
    max_pixel_error = 0
    
    for angle_deg in test_angles:
        angle_rad = angle_deg * np.pi / 180
        ray = np.array([[np.sin(angle_rad)], [0.0], [np.cos(angle_rad)]])
        
        uu_fish, vv_fish = fisheye.project_points_from_camera_to_image(ray)
        uu_ft, vv_ft = ftheta.project_points_from_camera_to_image(ray)
        
        if uu_fish[0] < 0 or uu_ft[0] < 0:
            print(f"  {angle_deg}°: out of FOV")
            continue
        
        pixel_error = np.sqrt((uu_fish[0] - uu_ft[0])**2 + (vv_fish[0] - vv_ft[0])**2)
        max_pixel_error = max(max_pixel_error, pixel_error)
        
        status = "✓" if pixel_error < 0.5 else "✗"
        print(f"  {angle_deg}°: fisheye=({uu_fish[0]:.1f},{vv_fish[0]:.1f}) "
              f"ftheta=({uu_ft[0]:.1f},{vv_ft[0]:.1f}) err={pixel_error:.4f}px {status}")
    
    print(f"Max pixel error: {max_pixel_error:.4f}")
    return max_pixel_error < 1.0


def test_motovis_real_image():
    """Test rendering a real motovis fisheye image using both camera models."""
    print("\n=== Test 11: Real Image Rendering ===")
    
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping")
        return True
    
    calib_path = '/home/alpha/Projects/VirtualCamera/data/motovis/calibration_fix/calibration_ipm_back.yml'
    img_path = '/home/alpha/Projects/VirtualCamera/data/motovis/camera/camera5/1703069369564.jpg'
    
    if not os.path.exists(calib_path) or not os.path.exists(img_path):
        print("Real data not found, skipping")
        return True
    
    with open(calib_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    camera_cfg = cfg['rig']['camera5']
    src_img = cv2.imread(img_path)
    
    if src_img is None:
        print("Failed to read image, skipping")
        return True
    
    print(f"Image shape: {src_img.shape}")
    
    # Create source cameras from same config
    fisheye_cam = FisheyeCamera.init_from_motovis_cfg(camera_cfg, use_default_fov=False)
    ftheta_cam = FThetaCamera.init_from_motovis_cfg(camera_cfg, use_default_fov=False)
    
    # Also test conversion
    ftheta_converted = convert_fisheye_to_ftheta(fisheye_cam)
    fisheye_converted = convert_ftheta_to_fisheye(ftheta_cam)
    
    # Create a perspective destination camera
    dst_camera = PerspectiveCamera(
        resolution=(1280, 960),
        extrinsic=(fisheye_cam.R_e, fisheye_cam.t_e),
        intrinsic=(640, 480, 640.0, 640.0)
    )
    
    # Render with each source camera
    dst_fish, mask_fish = render_image(src_img, fisheye_cam, dst_camera)
    dst_ftheta, mask_ftheta = render_image(src_img, ftheta_cam, dst_camera)
    dst_conv, mask_conv = render_image(src_img, ftheta_converted, dst_camera)
    
    # Compare rendered images
    valid_mask = (mask_fish > 0) & (mask_ftheta > 0)
    if np.sum(valid_mask) > 0:
        diff = np.abs(dst_fish.astype(float) - dst_ftheta.astype(float))
        mean_diff = np.mean(diff[np.stack([valid_mask]*3, axis=-1)])
        max_diff = np.max(diff[np.stack([valid_mask]*3, axis=-1)])
        
        print(f"Fisheye vs FTheta render difference:")
        print(f"  Mean pixel diff: {mean_diff:.4f}")
        print(f"  Max pixel diff:  {max_diff:.4f}")
        
        # Save comparison images
        cv2.imwrite('/tmp/render_fisheye.jpg', dst_fish)
        cv2.imwrite('/tmp/render_ftheta.jpg', dst_ftheta)
        cv2.imwrite('/tmp/render_converted.jpg', dst_conv)
        cv2.imwrite('/tmp/render_diff.jpg', (diff * 10).clip(0, 255).astype(np.uint8))
        print("  Saved to /tmp/render_*.jpg")
        
        # Difference should be very small (same underlying model)
        return mean_diff < 5.0  # Allow some interpolation difference
    else:
        print("No valid overlapping region, check FOV")
        return False


def test_motovis_all_cameras():
    """Test all cameras in the calibration file can be loaded as both models."""
    print("\n=== Test 12: All Motovis Cameras ===")
    
    calib_path = '/home/alpha/Projects/VirtualCamera/data/motovis/calibration_fix/calibration_ipm_back.yml'
    if not os.path.exists(calib_path):
        print("Calibration file not found, skipping")
        return True
    
    with open(calib_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    all_passed = True
    
    for cam_name, cam_cfg in cfg['rig'].items():
        if 'sensor_model' not in cam_cfg:
            continue
        if 'camera' not in cam_cfg.get('sensor_model', ''):
            continue
        
        try:
            fisheye = FisheyeCamera.init_from_motovis_cfg(cam_cfg, use_default_fov=False)
            ftheta = FThetaCamera.init_from_motovis_cfg(cam_cfg, use_default_fov=False)
            
            # Test round-trip at a moderate angle
            angle_rad = 30 * np.pi / 180
            ray = np.array([[np.sin(angle_rad)], [0.0], [np.cos(angle_rad)]])
            
            uu_fish, vv_fish = fisheye.project_points_from_camera_to_image(ray)
            uu_ft, vv_ft = ftheta.project_points_from_camera_to_image(ray)
            
            if uu_fish[0] >= 0 and uu_ft[0] >= 0:
                pixel_error = np.sqrt((uu_fish[0] - uu_ft[0])**2 + (vv_fish[0] - vv_ft[0])**2)
                status = "✓" if pixel_error < 0.5 else "✗"
                print(f"  {cam_name} (fov={cam_cfg.get('fov_fit', '?')}): "
                      f"err={pixel_error:.4f}px {status}")
                if pixel_error >= 0.5:
                    all_passed = False
            else:
                print(f"  {cam_name}: out of FOV at 30°")
                
        except Exception as e:
            print(f"  {cam_name}: FAILED - {type(e).__name__}: {e}")
            all_passed = False
    
    return all_passed


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("FThetaCamera Test Suite (Corrected Kannala-Brandt Model)")
    print("=" * 60)
    
    tests = [
        ("Forward Only", test_forward_only),
        ("Backward Only", test_backward_only),
        ("Both Polynomials", test_both_polynomials),
        ("NVIDIA Format", test_nvidia_format),
        ("FOV Computation", test_fov_computation),
        ("Polynomial Math", test_polynomial_math),
        ("Fisheye→FTheta", test_fisheye_to_ftheta_conversion),
        ("FTheta→Fisheye", test_ftheta_to_fisheye_conversion),
        ("Refit Round-Trip", test_refit_roundtrip),
        ("Motovis Init", test_motovis_ftheta_init),
        ("Real Image Render", test_motovis_real_image),
        ("All Motovis Cameras", test_motovis_all_cameras),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            import traceback
            print(f"\n✗ Test '{name}' failed with exception:")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
