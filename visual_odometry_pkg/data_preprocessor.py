import glob
import cv2 as cv


class DataPreprocessor:
    """

    This class object will provide set of methods for utility functions such as frames to video conversion

    """

    def __init__(self):
        """

        """
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

        files = glob.glob(source+'*.PNG')

        frames_array = []
        for x in files:
            keyframes = cv.imread(x)
            frames_array.append(keyframes)

        height, width, layers = frames_array[0].shape
        size = (height, width)
        if file_format == 'DIVX':
            video_out = cv.VideoWriter(destination, cv.VideoWriter_fourcc(*'DIVX'), fps, size)
        elif file_format == 'XVID':
            video_out = cv.VideoWriter(destination, cv.VideoWriter_fourcc(*'XVID'), fps, size)
        elif file_format == 'MJPG':
            video_out = cv.VideoWriter(destination, cv.VideoWriter_fourcc(*'MJPG'), fps, size)
        else:
            print("Video Format provided not supported.")
            return

        for keyframe in frames_array:
            video_out.write(keyframe)
        video_out.release()


if __name__ == '__main__':
    msg = 'data preprocessor Module of Visual odometry package.'
    print(f'{msg}')
