import glob
import cv2 as cv
import numpy as np
from scipy.ndimage import map_coordinates as interp2


class DataPreprocessor:
    """

    This class object will provide set of methods for utility functions such as frames to video conversion

    """

    def __init__(self):
        pass

    @staticmethod
    def frames_to_video(source: str, destination: str, file_format: str, fps: int = 24) -> None:
        """
        If a data is presented in frames it can be converted into a video file.

        Args:
            fps: int
                number of frames per second for output video
            file_format: string
                file format specifier (supports )
            source: string
                source folder path
            destination: string
                desired destination folder path

        Returns: None
        """

        files = glob.glob(source + '*.png')

        frames_array = []
        for x in files:
            keyframes = cv.imread(x)
            frames_array.append(keyframes)

        height, width, layers = frames_array[0].shape
        size = (width, height)
        if file_format == 'DIVX' or file_format == 'divx':
            video_out = cv.VideoWriter(destination+'out.avi', cv.VideoWriter_fourcc(*'DIVX'), fps, size)
        elif file_format == 'X264' or file_format == 'x264' or file_format == 'mkv':
            video_out = cv.VideoWriter(destination+'out.mkv', cv.VideoWriter_fourcc(*'X264'), fps, size)
        elif file_format == 'MJPG' or file_format == 'mjpg' or file_format == 'mp4':
            video_out = cv.VideoWriter(destination+'out.mp4', cv.VideoWriter_fourcc(*'MJPG'), fps, size)
        else:
            print("Video Format provided not supported.")
            return

        for keyframe in frames_array:
            video_out.write(keyframe)
        video_out.release()

    @staticmethod
    def undistort_image(image: np.array, lut: np.array) -> np.array:
        """

        Takes an distorted image and undistort it.

        Args:
            lut: np.array
                Undistortion lookup table
            image: np.array
                input image

        Returns: np.array

        """
        reshaped_lut = lut[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
        undistorted = np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1)
                                            for channel in range(0, image.shape[2])]), 0, 3)
        return undistorted.astype(image.dtype)

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
        return cv.cvtColor(bayer_image, cv.COLOR_BAYER_GR2BGR)


if __name__ == '__main__':
    msg = 'data preprocessor Module of Visual odometry package.'
    print(f'{msg}')
