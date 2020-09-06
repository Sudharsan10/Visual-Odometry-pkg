# ==================================================================================================================== #
# --------------------> Project Information <----------------------- #
# ==================================================================================================================== #
# Author(s)     :-> Sudharsan
# E-mail        :-> sudharsansci@gmail.com
# Project       :-> Visual odometry pkg
# URL           :-> http://iamsudharsan.com/Visual-Odometry-pkg/
# Module Desc   :-> Data preprocessor module for visual_odometry_pkg
# Description   :->

# ==================================================================================================================== #
# Import Section
# ==================================================================================================================== #
import glob
import logging
import cv2 as cv
import numpy as np
import concurrent.futures as cf
from collections import Iterator
from scipy.ndimage import map_coordinates as interp2

# ==================================================================================================================== #
# Logger setup section
# ==================================================================================================================== #
log = logging.getLogger(__name__)

file_handler = logging.FileHandler('./logs/data_preprocessor.log')
log_formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s :->%(message)s')

stream_formatter = logging.Formatter('%(levelname)s: %(name)s :->%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(stream_formatter)

log.addHandler(file_handler)
log.addHandler(stream_handler)

log.setLevel(logging.DEBUG)

log.info('\n# -------------------------------- #')
log.info('# -----> *** New Record *** <----- #')
log.info('# -------------------------------- #')

file_handler.setFormatter(log_formatter)


# ==================================================================================================================== #
# Class Definition
# ==================================================================================================================== #
class DataPreprocessor:
    """

    This class object will provide set of methods for utility functions such as frames to video conversion

    """

    def __init__(self):
        pass

    @staticmethod
    def undistort_image(image: np.array, lut: np.array) -> np.array:
        """

        Takes an distorted image and undistort it.

        Args:
            lut: np.array
                Undistortion lookup table
            image: np.array
                input color image of shape (m, n, 3)

        Returns: np.array

        """
        log.debug(f' DataPreprocessor.undistort_image() invoked..!')
        reshaped_lut = lut[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
        undistorted = np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1)
                                            for channel in range(0, image.shape[2])]), 0, 3)
        log.debug(f' Exiting undistort_image()..!')
        return undistorted.astype(image.dtype)

    @staticmethod
    def load_frames(files: list) -> Iterator:
        """
        Loads the frames from the files list
        Args:
            files: list
                list of filenames
        Returns: list
            Returns a list of keyframes
        """
        log.debug(f' DataProcessor.load_frames() invoked..!')
        with cf.ThreadPoolExecutor() as executor:
            frames_array = executor.map(lambda x: cv.imread(x, 0), files)
        log.debug(f' Exiting load_frames() successfully..!')
        return frames_array

    @staticmethod
    def frames_to_video(source: str, destination: str, file_format: str = 'DIVX', fps: int = 12) -> None:
        """
        If a data is presented in frames it can be converted into a video file.

        Args:
            fps: int
                number of frames per second for output video
            file_format: string
                file format specifier (supports .divx alone at the moment)
            source: string
                source folder path
            destination: string
                desired destination folder path

        Returns: None
        """
        log.debug(f' DataPreprocessor.frames_to_video() invoked..!')
        # ---> Step 01: Load file names and extract image details <--- #
        files = sorted(glob.glob(source + '*.png'))
        height, width = cv.imread(files[0], 0).shape
        size = (width, height)
        log.debug(f' Step 01: Completed Successfully..!')

        # ---> Step 02: Setup the video writer <--- #
        if file_format == 'DIVX' or file_format == 'divx':
            video_out = cv.VideoWriter(destination + 'out.avi', cv.VideoWriter_fourcc(*'DIVX'), fps, size)
        elif file_format == 'X264' or file_format == 'x264' or file_format == 'mkv':
            video_out = cv.VideoWriter(destination + 'out.mkv', cv.VideoWriter_fourcc(*'X264'), fps, size)
        elif file_format == 'MJPG' or file_format == 'mjpg' or file_format == 'mp4':
            video_out = cv.VideoWriter(destination + 'out.mp4', cv.VideoWriter_fourcc(*'MJPG'), fps, size)
        else:
            log.info('Video Format provided is not supported at the moment.')
            return
        log.debug(f' Step 02: Completed Successfully..!')

        # ---> Step 03: Load the Images using threading <--- #
        with cf.ThreadPoolExecutor() as executor:
            frames_array = executor.map(lambda x: cv.cvtColor(cv.imread(x, 0), cv.COLOR_BAYER_GR2BGR), files)
        log.debug(f' Step 03: Completed Successfully..!')

        # ---> Step 04: Convert and write frames to video <--- #
        for keyframe in frames_array:
            video_out.write(keyframe)
        video_out.release()
        log.debug(f' Step 04: Completed Successfully..!')
        log.debug(f' Exiting DataPreprocessor.frames_to_video()..!')

    @staticmethod
    def convert_bayer_bg2bgr(bayer_image: np.array) -> np.array:
        """
        Convert Bayer BG image to bgr
        Args:
            bayer_image: np.array
                input bayer image
        Returns:
            BGR Image
        """
        log.debug(f' DataPreprocessor.convert_bayer_bg2bgr() invoked..!')
        return cv.cvtColor(bayer_image, cv.COLOR_BAYER_BG2BGR)

    @staticmethod
    def convert_bayer_gb2bgr(bayer_image: np.array) -> np.array:
        """
        Convert Bayer GB image to bgr
        Args:
            bayer_image: np.array
                input bayer array

        Returns:
            BGR Image
        """
        log.debug(f' DataPreprocessor.convert_bayer_gb2bgr() invoked..!')
        return cv.cvtColor(bayer_image, cv.COLOR_BAYER_GB2BGR)

    @staticmethod
    def convert_bayer_rg2bgr(bayer_image: np.array) -> np.array:
        """
        Convert Bayer RG image to bgr
        Args:
            bayer_image: np.array
                input bayer array

        Returns:
            BGR Image
        """
        log.debug(f' DataPreprocessor.convert_bayer_RG2bgr() invoked..!')
        return cv.cvtColor(bayer_image, cv.COLOR_BAYER_RG2BGR)

    @staticmethod
    def convert_bayer_gr2bgr(bayer_image: np.array) -> np.array:
        """
        Convert Bayer GR image to bgr
        Args:
            bayer_image: np.array
                input bayer array

        Returns:
            BGR Image
        """
        log.debug(f' DataPreprocessor.convert_bayer_GR2bgr() invoked..!')
        return cv.cvtColor(bayer_image, cv.COLOR_BAYER_GR2BGR)


if __name__ == '__main__':
    msg = 'data preprocessor Module of Visual odometry package.'
    print(f'{msg}')
