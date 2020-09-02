from .camera import Camera
from .visualizer import Visualize
from .data_preprocessor import DataPreprocessor


class ImplementVO:
    """
    A default implementation pipeline for VO
    """

    def __init__(self, camera: object, data_processor: object, visualize: object):
        self.cam = Camera()
        self.data_preprocessor = DataPreprocessor()
        self.visualize = Visualize()

    def import_data(self, cam_model_dir: str, data_dir: str) -> None:
        """
        Import camera model and data from the designated folder
        Args:
            cam_model_dir: str
                Path to the camera model dir
            data_dir: str
                Path to the image frames dir

        Returns:

        """

        self.cam.read_camera_model(cam_model_dir)
        print('Camera Model Read Successfully...!')


if __name__ == '__main__':
    msg = 'handler Module of Visual odometry package.'
    print(f'{msg}')
