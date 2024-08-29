from typing import Literal
import cv2
import numpy as np


def frame_loader():
    pass

class PreprocessFrame:
    @staticmethod
    def _canny(frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur_frame = cv2.GaussianBlur(gray_frame, (5,5), 0)
        canny_frame = cv2.Canny(blur_frame, 50, 150)

        return canny_frame

    @staticmethod
    def _region_of_interest(frame):
        height = frame.shape[0]
        polygons = np.array([[(200, height), (1100, height), (550, 250)]])
        mask = np.zeros_like(frame)

        cv2.fillPoly(mask, polygons, color = 255)

        masked_frame = cv2.bitwise_and(frame, mask)

        return masked_frame

    @staticmethod
    def _make_coordinates(frame, line_parameters):
        slope, intercept = line_parameters
        y1 = frame.shape[0]
        y2 = int(y1*(3/5))
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)

        return np.array([x1, y1, x2, y2])

    def _average_slope_intercept(self, frame, lines):
        left_fit = []
        right_fit = []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
 
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)

        left_line = self._make_coordinates(frame, left_fit_average)
        right_line = self._make_coordinates(frame, right_fit_average)


        return np.array([left_line, right_line])

    @staticmethod
    def _overlay_lines_on_frame(frame, lines):
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        return line_image

    def _find_lines(self, cropped_frame, original_frame):
        lines = cv2.HoughLinesP(cropped_frame, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = self._average_slope_intercept(original_frame, lines)
        line_image = self._overlay_lines_on_frame(original_frame, averaged_lines)

        return line_image

    def preprocess(self, frame):
        copied_frame = np.copy(frame)
        canny_frame = self._canny(copied_frame)    
        cropped_frame = self._region_of_interest(canny_frame)
        lines_frame = self._find_lines(cropped_frame, copied_frame)
        combined_frame = cv2.addWeighted(copied_frame, 0.8, lines_frame, 1.0, 1.0)

        return combined_frame

class LaneFinder:
    def __init__(self, frames, input_type: Literal["image", "video"]) -> None:
        self.frames = frames
        self.input_type = input_type
        self.preprocess_frame = PreprocessFrame()

    @classmethod
    def from_image(cls, image_path:str):
        img = cv2.imread(image_path)
        return cls(frames = img, input_type="image")

    @classmethod
    def from_video(cls, video_path:str):
        cap = cv2.VideoCapture(video_path)

        if (cap.isOpened()== False):
            print("Error opening video file")
                
        return cls(frames=cap, input_type="video")

    def run(self, *args, **kwargs):
        
        if self.input_type == "image":
            processed_frame = self.preprocess_frame.preprocess(self.frames)
            cv2.imshow("result", processed_frame)
            cv2.waitKey(0)
        elif self.input_type == "video":
          
            while self.frames.isOpened():
                ret, frame = self.frames.read()
                if ret == True:
                    processed_frame = self.preprocess_frame.preprocess(frame)
                    cv2.imshow("result", processed_frame)
                    cv2.waitKey(10)
                else:
                    break

def test_from_img():
    lane_finder = LaneFinder.from_image("./resources/test_image.jpg")
    lane_finder.run()

def test_from_vid():
    lane_finder = LaneFinder.from_video("./resources/test.mp4")
    lane_finder.run()

if __name__ == "__main__":
    test_from_vid()