# ==================================================================================================================== #
# --------------------> Project Information <----------------------- #
# ==================================================================================================================== #
# Author(s)     :-> Sudharsan
# E-mail        :-> sudharsansci@gmail.com
# Project       :-> Visual odometry pkg
# URL           :-> http://iamsudharsan.com/Visual-Odometry-pkg/
# Module Desc   :-> Test Module for camera.py module
# Description   :->

# ==================================================================================================================== #
# Import Section
# ==================================================================================================================== #
import unittest
import logging
import numpy as np
from visual_odometry_pkg import camera

# ==================================================================================================================== #
# Logger setup section
# ==================================================================================================================== #
log = logging.getLogger(__name__)

file_handler = logging.FileHandler('./logs/test_camera.log')
log_formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s :->%(message)s')

stream_formatter = logging.Formatter('%(levelname)s: %(name)s :->%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(stream_formatter)

log.addHandler(file_handler)
log.addHandler(stream_handler)

log.setLevel(logging.INFO)

log.info('\n# -------------------------------- #')
log.info('# -----> *** New Record *** <----- #')
log.info('# -------------------------------- #')

file_handler.setFormatter(log_formatter)

# ==================================================================================================================== #
# Global Variable Section
# ==================================================================================================================== #
test_data = '../test_data/'
source = '../test_data/'
destination = '../test_data/out/'


# ==================================================================================================================== #
# Class Definition
# ==================================================================================================================== #
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

        Test Condition:
            Input   :-> intrinsic.txt and lut.bin files
            Output  :-> fx, fy, cx, cy, G_camera_image, LUT values
        """
        self.cam.read_camera_model(test_data)
        # verify fx, fy, cx, cy
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
        log.info(f' read_camera_model() passed')

    def test_get_camera_model(self) -> None:
        """
        Testing getter function for camera model parameters

        Test Condition:
            Output  :-> returns (fx, fy, cx, cy, G_camera_image, LUT) tuple of shape == (6,)
        """
        self.assertEqual(self.cam.get_camera_model().__len__(), 6)
        log.info(f' get_camera_model() passed')


if __name__ == '__main__':
    unittest.main()