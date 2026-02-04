"""
Test suite for FThetaCamera implementation.
Tests round-trip consistency, polynomial validation, and NVIDIA dataset compatibility.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/alpha/Projects/VirtualCamera')

from virtual_camera import FThetaCamera
from scipy.spatial.transform import Rotation


def test_forward_only():
    """Test FThetaCamera with forward polynomial only."""
    print("\n=== Test 1: Forward Only ===")
    
    # Create camera with forward coefficients
    # Format: (cx, cy, fx, fy, 'F', k1, k2, k3, k4, k5)
    intrinsic = (960, 540, 800.0, 800.0, 'F', 
                 800.0, 0.1, -0.001, 0.0001, 0.0)
    
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
    
    # Test round-trip
    test_passed = test_round_trip(camera)
    return test_passed


def test_backward_only():
    """Test FThetaCamera with backward polynomial only (NVIDIA format)."""
    print("\n=== Test 2: Backward Only (NVIDIA Format) ===")
    
    # Create camera with backward coefficients
    # Format: (cx, cy, fx, fy, 'B', j1, j2, j3, j4, j5)
    intrinsic = (960, 540, 800.0, 800.0, 'B',
                 0.00125, 0.0, 0.0, 0.0, 0.0)
    
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
    
    # Test round-trip
    test_passed = test_round_trip(camera)
    return test_passed


def test_both_polynomials():
    """Test FThetaCamera with both forward and backward polynomials."""
    print("\n=== Test 3: Both Polynomials ===")
    
    # Create camera with both polynomials
    # Format: (cx, cy, fx, fy, 'FB', k1..k5, j1..j5)
    forward_coeffs = [800.0, 0.1, -0.001, 0.0001, 0.0]
    backward_coeffs = [0.00125, 0.0, 0.0, 0.0, 0.0]
    
    intrinsic = (960, 540, 800.0, 800.0, 'FB') + tuple(forward_coeffs) + tuple(backward_coeffs)
    
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
    
    # Test round-trip
    test_passed = test_round_trip(camera)
    
    # Validate polynomial consistency
    print("\nValidating polynomial consistency...")
    theta_test = np.linspace(0.01, np.pi/3, 100)
    r_forward = camera._evaluate_polynomial(theta_test, camera.forward_coeffs)
    theta_recovered = camera._evaluate_polynomial(r_forward, camera.backward_coeffs)
    max_error = np.max(np.abs(theta_recovered - theta_test))
    print(f"Max round-trip error: {max_error:.6f} radians")
    
    if max_error < 1e-3:
        print("✓ Polynomial pair is consistent")
        return test_passed
    else:
        print("⚠ Polynomial pair has significant error")
        return False


def test_nvidia_format():
    """Test initialization from NVIDIA dataset format."""
    print("\n=== Test 4: NVIDIA Dataset Format ===")
    
    # Simulate NVIDIA parquet row
    nvidia_cfg = {
        'width': 1920,
        'height': 1080,
        'cx': 960.0,
        'cy': 540.0,
        'bw_poly_0': 0.00125,
        'bw_poly_1': 0.0,
        'bw_poly_2': 0.0,
        'bw_poly_3': 0.0,
        'bw_poly_4': 0.0,
        'fw_poly_0': 800.0,
        'fw_poly_1': 0.1,
        'fw_poly_2': -0.001,
        'fw_poly_3': 0.0001,
        'fw_poly_4': 0.0,
    }
    
    camera = FThetaCamera.init_from_nvidia_cfg(nvidia_cfg)
    
    print(f"Resolution: {camera.resolution}")
    print(f"Principal point: ({camera.cx}, {camera.cy})")
    print(f"Forward coeffs: {camera.forward_coeffs}")
    print(f"Backward coeffs: {camera.backward_coeffs}")
    
    # Test round-trip
    test_passed = test_round_trip(camera)
    return test_passed


def test_round_trip(camera):
    """
    Test round-trip consistency: camera -> image -> camera
    Uses exact pixel coordinates for precise testing.
    """
    print("\nTesting round-trip consistency...")
    
    all_passed = True
    max_error = 0
    
    # Test angles within reasonable range
    for angle_deg in [-60, -45, -30, 0, 30, 45, 60]:
        angle_rad = angle_deg * np.pi / 180
        
        # Create test ray at this angle (in camera coordinates)
        test_ray = np.array([
            [np.sin(angle_rad)],  # x
            [0.0],                 # y
            [np.cos(angle_rad)]    # z
        ])
        
        # Project to image
        uu, vv = camera.project_points_from_camera_to_image(test_ray)
        
        # Check if projection is valid
        if uu[0] < 0 or vv[0] < 0 or uu[0] >= camera.resolution[0] or vv[0] >= camera.resolution[1]:
            print(f"  Angle {angle_deg}°: Out of FOV")
            continue
        
        # For round-trip test, we need to unproject the exact pixel coordinate
        # Get the ray direction from unprojection at this pixel
        W, H = camera.resolution
        u_pixel = int(round(uu[0]))
        v_pixel = int(round(vv[0]))
        
        # Clamp to valid range
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
        
        if angular_error < 1.0:  # Less than 1 degree error
            print(f"  Angle {angle_deg}°: ✓ Error = {angular_error:.4f}°")
        else:
            print(f"  Angle {angle_deg}°: ✗ Error = {angular_error:.4f}° (too large)")
            all_passed = False
    
    print(f"Max angular error: {max_error:.4f}°")
    return all_passed and max_error < 1.0


def test_fov_computation():
    """Test automatic FOV computation."""
    print("\n=== Test 5: FOV Computation ===")
    
    # Create camera without explicit FOV
    intrinsic = (960, 540, 800.0, 800.0, 'F',
                 800.0, 0.1, -0.001, 0.0001, 0.0)
    
    R = np.eye(3)
    t = np.zeros(3)
    
    camera = FThetaCamera(
        resolution=(1920, 1080),
        extrinsic=(R, t),
        intrinsic=intrinsic
        # fov=None -> auto-compute
    )
    
    print(f"Auto-computed FOV: {camera.fov:.2f}°")
    print(f"✓ FOV computation works" if camera.fov > 0 else "✗ FOV computation failed")
    
    return camera.fov > 0


def test_visual_example():
    """Create a visual example of FTHETA camera projection."""
    print("\n=== Test 6: Visual Example ===")
    
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping visual test")
        return True
    
    # Create source FTHETA camera (e.g., fisheye)
    src_intrinsic = (960, 540, 800.0, 800.0, 'F',
                     600.0, 0.2, -0.002, 0.0001, 0.0)
    
    R_src = np.eye(3)
    t_src = np.zeros(3)
    
    src_camera = FThetaCamera(
        resolution=(1920, 1080),
        extrinsic=(R_src, t_src),
        intrinsic=src_intrinsic,
        fov=180
    )
    
    # Create destination perspective camera
    from virtual_camera import PerspectiveCamera
    
    dst_intrinsic = (640, 360, 800.0, 800.0)
    dst_camera = PerspectiveCamera(
        resolution=(1280, 720),
        extrinsic=(R_src, t_src),
        intrinsic=dst_intrinsic
    )
    
    # Create synthetic test image
    src_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Draw a grid
    for i in range(0, 1920, 100):
        cv2.line(src_img, (i, 0), (i, 1080), (255, 255, 255), 1)
    for i in range(0, 1080, 100):
        cv2.line(src_img, (0, i), (1920, i), (255, 255, 255), 1)
    
    # Draw a circle at center
    cv2.circle(src_img, (960, 540), 200, (0, 0, 255), 2)
    cv2.circle(src_img, (960, 540), 400, (0, 255, 0), 2)
    
    # Render to perspective view
    from virtual_camera import render_image
    dst_img, dst_mask = render_image(src_img, src_camera, dst_camera)
    
    # Save example images
    cv2.imwrite('/tmp/ftheta_src.jpg', src_img)
    cv2.imwrite('/tmp/ftheta_dst.jpg', dst_img)
    cv2.imwrite('/tmp/ftheta_mask.jpg', (dst_mask * 255).astype(np.uint8))
    
    print("✓ Visual example saved to /tmp/ftheta_*.jpg")
    print("  - ftheta_src.jpg: Source FTHETA view with grid")
    print("  - ftheta_dst.jpg: Destination perspective view")
    print("  - ftheta_mask.jpg: Valid region mask")
    
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("FThetaCamera Test Suite")
    print("=" * 60)
    
    tests = [
        ("Forward Only", test_forward_only),
        ("Backward Only", test_backward_only),
        ("Both Polynomials", test_both_polynomials),
        ("NVIDIA Format", test_nvidia_format),
        ("FOV Computation", test_fov_computation),
        ("Visual Example", test_visual_example),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception:")
            print(f"  {type(e).__name__}: {e}")
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
