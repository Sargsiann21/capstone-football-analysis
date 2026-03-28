import cv2
import numpy as np


class TeamStructureDrawer:

    def __init__(self, k_neighbors=3):
        self.k = k_neighbors
        self.field_line_alpha = 0.25


    def get_foot_position(self, bbox):
        x1, y1, x2, y2 = bbox
        x = int((x1 + x2) / 2)
        y = int(y2)
        return (x, y)


    def draw_team_structure(self, frame, player_tracks, frame_num):

        team1_positions = []
        team2_positions = []

        for player_id, player in player_tracks[frame_num].items():

            bbox = player["bbox"]
            team = player.get("team", None)

            if team is None:
                continue

            position = self.get_foot_position(bbox)

            if team == 1:
                team1_positions.append(position)

            elif team == 2:
                team2_positions.append(position)

        frame = self._connect_players(frame, team1_positions, (0,0,255))
        frame = self._connect_players(frame, team2_positions, (255,0,0))

        return frame


    def _connect_players(self, frame, positions, color):

        if len(positions) < 2:
            return frame

        positions = np.array(positions)
        overlay = frame.copy()

        for i in range(len(positions)):

            distances = np.linalg.norm(positions - positions[i], axis=1)

            nearest_indices = np.argsort(distances)[1:self.k+1]

            for j in nearest_indices:

                p1 = tuple(positions[i].astype(int))
                p2 = tuple(positions[j].astype(int))

                cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

            cv2.addWeighted(
                overlay,
                self.field_line_alpha,
                frame,
                1 - self.field_line_alpha,
                0,
                frame
            )

        return frame


    def draw_structure_panel(self, frames, tracks):

        for frame_num, frame in enumerate(frames):

            team1_positions = []
            team2_positions = []

            for player_id, player in tracks["Players"][frame_num].items():

                bbox = player["bbox"]
                team = player.get("team", None)

                position = self.get_foot_position(bbox)

                if team == 1:
                    team1_positions.append(position)

                elif team == 2:
                    team2_positions.append(position)

            h, w, _ = frame.shape

            panel_x1 = w - 300
            panel_x2 = w - 20
            panel_y1 = 120
            panel_y2 = 300

            cv2.rectangle(frame,(panel_x1,panel_y1),(panel_x2,panel_y2),(255,255,255),-1)

            cv2.putText(frame,"Team Structures",(panel_x1+20,panel_y1+25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

            self._draw_mini_structure(frame, team1_positions,
                                      panel_x1+20, panel_y1+40, (0,0,255))

            self._draw_mini_structure(frame, team2_positions,
                                      panel_x1+20, panel_y1+120, (255,0,0))


    def _draw_mini_structure(self, frame, positions, x_offset, y_offset, color):

        if len(positions) == 0:
            return

        positions = np.array(positions)

        min_x = np.min(positions[:,0])
        max_x = np.max(positions[:,0])

        min_y = np.min(positions[:,1])
        max_y = np.max(positions[:,1])

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        mini_points = []

        for p in positions:

            x = int((p[0]-min_x)/width * 200) + x_offset
            y = int((p[1]-min_y)/height * 60) + y_offset

            mini_points.append((x, y))

        if len(mini_points) >= 2:

            mini_points_np = np.array(mini_points, dtype=np.float32)
            max_neighbors = min(self.k, len(mini_points) - 1)
            drawn_edges = set()

            for i in range(len(mini_points_np)):

                distances = np.linalg.norm(mini_points_np - mini_points_np[i], axis=1)
                nearest_indices = np.argsort(distances)[1:max_neighbors + 1]

                for j in nearest_indices:

                    edge = tuple(sorted((i, int(j))))

                    if edge in drawn_edges:
                        continue

                    drawn_edges.add(edge)
                    p1 = tuple(mini_points_np[i].astype(int))
                    p2 = tuple(mini_points_np[j].astype(int))

                    cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

        for x, y in mini_points:

            cv2.circle(frame,(x,y),4,color,-1)