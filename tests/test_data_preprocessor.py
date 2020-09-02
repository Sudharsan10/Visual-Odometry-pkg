import unittest
import cv2 as cv
import os
from visual_odometry_pkg import data_preprocessor as dp


class TestDataPreprocessor(unittest.TestCase):
    """
    Test class for Data preprocessor class
    """

    def setUp(self) -> None:
        """
        Create an object of DataPreprocessor
        """
        self.data_preprocessor = dp.DataPreprocessor()
        self.bayer = cv.imread('../test_data/1.png', 0)

    def test_frames_to_video(self) -> None:
        """
        Test the frames to video function for output
        """
        source = '../test_data/'
        destination = '../test_data/out/'

        if os.path.isdir(destination):
            if os.path.isfile(destination+'out.avi'):
                os.remove(destination+'out.avi')
            self.data_preprocessor.frames_to_video(source, destination, 'DIVX', 1)
        else:
            os.makedirs(destination)
            self.data_preprocessor.frames_to_video(source, destination, 'DIVX', 1)

        self.assertTrue(os.path.isfile(destination+'out.avi'))

        # if os.path.isdir(destination):
        #     if os.path.isfile(destination+'out.mkv'):
        #         os.remove(destination+'out.mkv')
        #     self.data_preprocessor.frames_to_video(source, destination, 'X264', 1)
        # else:
        #     os.makedirs(destination)
        #     self.data_preprocessor.frames_to_video(source, destination, 'X264', 1)
        # self.assertTrue(os.path.isfile(destination+'out.mkv'))

    def test_convert_bayer_gr2bgr(self) -> None:
        """
        Converts a bayer image into color image.
        input: (x,x) -> output: (x,x, 3)
        """
        result = self.data_preprocessor.convert_bayer_gr2bgr(self.bayer)
        self.assertEqual(result.shape[2], 3)

    def test_convert_bayer_bg2bgr(self) -> None:
        """
        Converts a bayer image into color image.
        input: (x,x) -> output: (x,x, 3)
        """
        result = self.data_preprocessor.convert_bayer_bg2bgr(self.bayer)
        self.assertEqual(result.shape[2], 3)

    def test_convert_bayer_gb2bgr(self) -> None:
        """
        Converts a bayer image into color image.
        input: (x,x) -> output: (x,x, 3)
        """
        result = self.data_preprocessor.convert_bayer_gb2bgr(self.bayer)
        self.assertEqual(result.shape[2], 3)

    def test_convert_bayer_rg2bgr(self) -> None:
        """
        Converts a bayer image into color image.
        input: (x,x) -> output: (x,x, 3)
        """
        result = self.data_preprocessor.convert_bayer_rg2bgr(self.bayer)
        self.assertEqual(result.shape[2], 3)
