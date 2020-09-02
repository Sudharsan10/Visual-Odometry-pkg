import unittest
import numpy as np
from visual_odometry_pkg import camera


class TestCamera(unittest.TestCase):
    """
        Test class for camera module in Visual odometry package
    """

    def setUp(self) -> None:
        self.cam = camera.Camera()

    def test_read_camera_model(self) -> None:
        """
        Testing read_camera_model().

        loads camera model from test data folder and verifies its value.
        Basically it an fetch test.
        """
        self.cam.read_camera_model('../test_data/')
        # fx, fy, cx, cy
        self.assertEqual(self.cam.fx, 964.828979)
        self.assertEqual(self.cam.fy, 964.828979)
        self.assertEqual(self.cam.cx, 643.788025)
        self.assertEqual(self.cam.cy, 484.40799)

        try:
            np.testing.assert_array_almost_equal(np.array([[0., -0., 1., 0.],
                                                           [1., 0., -0., 0.],
                                                           [0., 1., 0., 0.],
                                                           [0., 0., 0., 1.]]), self.cam.G_camera_image)

            g_camera_image = True
        except AssertionError as err:
            g_camera_image = False
        self.assertTrue(g_camera_image)

    def test_get_camera_model(self) -> None:
        """
        Testing getter function for camera model parameters
        """
        self.assertEqual(self.cam.get_camera_model().__len__(), 6)
