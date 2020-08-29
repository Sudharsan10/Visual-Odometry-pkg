from typing import Any, Tuple
import numpy as np


class Camera:
    """



    """

    def __init__(self):
        """

        Camera model information variables initialization

        """
        self.fx = None  # horizontal focal length in pixels
        self.fy = None  # vertical focal length in pixels
        self.cx = None  # horizontal principal point in pixels
        self.cy = None  # vertical principal point in pixels
        self.LUT = None  # undistortion lookup table
        self.G_camera_image = None

    def read_camera_model(self, model_dir: str, intrinsic_filename: str = 'intrinsic_parameters.txt',
                          lut_filename: str = 'intrinsic_parameters.txt') -> Tuple[Any, Any, Any, Any, Any, Any]:
        """

        Args:
            lut_filename: string
            intrinsic_filename: string
            model_dir: string
                Path to the model folder from root of the project.

        Returns: list
            returns a list of internal camera parameters
                fx : horizontal focal length in pixels
                fy : vertical focal length in pixels
                cx : horizontal principal point in pixels
                cy : vertical principal point in pixels

                G_camera_image : transform that maps from image coordinates to the base frame of the camera. For
                monocular cameras, this is simply a rotation. For stereo camera, this is a rotation and translation to
                the left-most lens.

                LUT: undistortion lookup table. For an image of size w x h, LUT will be an array of size [w x h, 2],
                with a (u,v) pair for each pixel. Maps pixels in the undistorted image to pixels in the distorted image.

        """

        # ---> Read Intrinsic Parameters <--- #
        intrinsics = np.loadtxt(model_dir + intrinsic_filename)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[0, 1]
        self.cx = intrinsics[0, 2]
        self.cy = intrinsics[0, 3]

        # 4x4 matrix that transforms x-forward coordinate frames at camera origin and image frame for specific lens
        self.G_camera_image = intrinsics[1:5, 0:4]

        # ---> Read LUT for undistort the image <--- #
        lut = np.fromfile(model_dir + lut_filename, np.double)
        lut = lut.reshape([2, lut.size // 2])
        self.LUT = lut.T

        return self.fx, self.fy, self.cx, self.cy, self.G_camera_image, self.LUT


if __name__ == '__main__':
    msg = 'Camera Module of Visual odometry package.'
    print(f'{msg}')