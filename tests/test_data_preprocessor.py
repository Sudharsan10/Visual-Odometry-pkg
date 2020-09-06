# ==================================================================================================================== #
# --------------------> Project Information <----------------------- #
# ==================================================================================================================== #
# Author(s)     :-> Sudharsan
# E-mail        :-> sudharsansci@gmail.com
# Project       :-> Visual odometry pkg
# URL           :-> http://iamsudharsan.com/Visual-Odometry-pkg/
# Module Desc   :-> Test Module for data_preprocessor.py module
# Description   :->

# ==================================================================================================================== #
# Import Section
# ==================================================================================================================== #
import os
import logging
import unittest
import cv2 as cv
from visual_odometry_pkg import data_preprocessor as dp
from visual_odometry_pkg import camera

# ==================================================================================================================== #
# Logger setup section
# ==================================================================================================================== #
log = logging.getLogger(__name__)

file_handler = logging.FileHandler('./logs/test_data_preprocessor.log')
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
class TestDataPreprocessor(unittest.TestCase):
    """
    Test class for Data preprocessor class
    """

    def setUp(self) -> None:
        """
        Create an object of DataPreprocessor
        """
        self.cam = camera.Camera()
        self.cam.read_camera_model(test_data)
        self.bayer = cv.imread(test_data + '1.png', 0)
        self.data_preprocessor = dp.DataPreprocessor()

    def test_undistort_image(self) -> None:
        """
        Test the undistort image function

        Test Condition:
            Input   :-> Takes an bayer image
            Output  :-> Returns an image of shape(m, n, 3)
        """

        image = self.data_preprocessor.undistort_image(cv.cvtColor(self.bayer, cv.COLOR_BAYER_GR2BGR), self.cam.LUT)
        self.assertEqual(3, image.shape[2])
        log.info(f' undistort_image() passed!')

    def test_frames_to_video(self) -> None:
        """
        Test the frames to video function for output

        Test Condition:
            Input   :-> source dir for sample bayer images, destination dir for video out, desired file format, fps
            Output  :-> out.avi in the destination dir
        """

        if os.path.isdir(destination):
            if os.path.isfile(destination + 'out.avi'):
                os.remove(destination + 'out.avi')
            self.data_preprocessor.frames_to_video(source, destination, 'DIVX', 1)
        else:
            os.makedirs(destination)
            self.data_preprocessor.frames_to_video(source, destination, 'DIVX', 1)

        self.assertTrue(os.path.isfile(destination + 'out.avi'))
        log.info(f' frames_to_video() passed!')

    def test_convert_bayer_gr2bgr(self) -> None:
        """
        Converts a bayer image into color image.

        Test Condition:
            input   :-> (x,x)
            output  :-> (x,x, 3)
        """
        result = self.data_preprocessor.convert_bayer_gr2bgr(self.bayer)
        self.assertEqual(result.shape[2], 3)
        log.info(f' convert_bayer_gr2bgr() passed!')

    def test_convert_bayer_bg2bgr(self) -> None:
        """
        Converts a bayer image into color image.

        Test Condition:
            input   :-> (x,x)
            output  :-> (x,x, 3)

        """
        result = self.data_preprocessor.convert_bayer_bg2bgr(self.bayer)
        self.assertEqual(result.shape[2], 3)
        log.info(f' convert_bayer_bg2bgr() passed!')

    def test_convert_bayer_gb2bgr(self) -> None:
        """
        Converts a bayer image into color image.

        Test Condition:
            input   :-> (x,x)
            output  :-> (x,x, 3)
        """
        result = self.data_preprocessor.convert_bayer_gb2bgr(self.bayer)
        self.assertEqual(result.shape[2], 3)
        log.info(f' convert_bayer_gb2bgr() passed!')

    def test_convert_bayer_rg2bgr(self) -> None:
        """
        Converts a bayer image into color image.

        Test Condition:
            input   :-> (x,x)
            output  :-> (x,x, 3)
        """
        result = self.data_preprocessor.convert_bayer_rg2bgr(self.bayer)
        self.assertEqual(result.shape[2], 3)
        log.info(f' convert_bayer_rg2bgr() passed!')


if __name__ == '__main__':
    unittest.main()
