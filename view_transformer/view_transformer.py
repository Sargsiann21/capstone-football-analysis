import numpy as np
import cv2
from ultralytics import YOLO
import pickle
import os

class ViewTransformer():

    def __init__(self, first_frame, read_from_stub=False, stub_path=None):

        self.first_frame = first_frame
        self.field_model = None
        self.pixel_vertices = None

        court_width = 68
        court_length = 105

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ], dtype=np.float32)

        self.perspective_transformer = None

        use_stub = read_from_stub and stub_path is not None and os.path.exists(stub_path)

        if not use_stub:
            self._initialize_transformer(self.first_frame)


    def _initialize_transformer(self, frame):

        if self.perspective_transformer is not None:
            return

        if self.field_model is None:
            self.field_model = YOLO("models/field_best.pt")

        pixel_vertices = self.detect_full_field(frame)

        if pixel_vertices is None:
            raise ValueError("Full football field could not be detected.")

        self.pixel_vertices = pixel_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices,
            self.target_vertices
        )


    def detect_full_field(self, frame):

        results = self.field_model(frame, conf=0.5)[0]

        if len(results.boxes) == 0:
            return None

        boxes = results.boxes.xyxy.cpu().numpy()

        largest_box = max(
            boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
        )

        x1, y1, x2, y2 = largest_box

        bottom_left = [x1, y2]
        top_left = [x1, y1]
        top_right = [x2, y1]
        bottom_right = [x2, y2]

        return np.array([
            bottom_left,
            top_left,
            top_right,
            bottom_right
        ])


    def transform_point(self, point):

        reshaped_point = np.array(point).reshape(-1, 1, 2).astype(np.float32)

        transformed = cv2.perspectiveTransform(
            reshaped_point,
            self.perspective_transformer
        )

        return transformed.reshape(-1, 2)[0]


    def _normalize_point(self, point):

        if point is None:
            return None

        point_arr = np.array(point, dtype=np.float32).reshape(-1, 2)

        if len(point_arr) == 0:
            return None

        return point_arr[0]



    def add_transformed_position_tracks(self, tracks, read_from_stub=False, stub_path=None):

        # ---- LOAD FROM STUB ----
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):

            with open(stub_path, "rb") as f:
                transformed_data = pickle.load(f)

            for frame_num in range(len(tracks["Players"])):

                for player_id in tracks["Players"][frame_num]:

                    tracks["Players"][frame_num][player_id]["position_transformed"] = \
                        self._normalize_point(transformed_data["players"][frame_num].get(player_id))

                if frame_num < len(tracks["Ball"]):

                    for ball_id in tracks["Ball"][frame_num]:

                        tracks["Ball"][frame_num][ball_id]["position_transformed"] = \
                            self._normalize_point(transformed_data["ball"][frame_num].get(ball_id))

            return

        self._initialize_transformer(self.first_frame)


        # ---- NORMAL CALCULATION ----
        transformed_data = {
            "players": [],
            "ball": []
        }

        for frame_num in range(len(tracks["Players"])):

            player_frame = {}
            ball_frame = {}

            for player_id, player in tracks["Players"][frame_num].items():

                position = player.get("position_adjusted")

                if position is None:
                    player_frame[player_id] = None
                    continue

                transformed = self.transform_point(position)

                tracks["Players"][frame_num][player_id]["position_transformed"] = transformed
                player_frame[player_id] = transformed


            for ball_id, ball in tracks["Ball"][frame_num].items():

                position = ball.get("position_adjusted")

                if position is None:
                    ball_frame[ball_id] = None
                    continue

                transformed = self.transform_point(position)

                tracks["Ball"][frame_num][ball_id]["position_transformed"] = transformed
                ball_frame[ball_id] = transformed


            transformed_data["players"].append(player_frame)
            transformed_data["ball"].append(ball_frame)


        # ---- SAVE STUB ----
        if stub_path is not None:

            os.makedirs(os.path.dirname(stub_path), exist_ok=True)

            with open(stub_path, "wb") as f:
                pickle.dump(transformed_data, f)