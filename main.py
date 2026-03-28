from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
import os
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement import CameraMovement
from view_transformer import ViewTransformer
from speed_distance import SpeedDistance
from team_structure import TeamStructureDrawer
from formation_detector import FormationDetector


def main():

    # -------------------------
    # Read video
    # -------------------------
    video_frames = read_video('input_videos/input_video.mp4')

    # -------------------------
    # Initialize tracker
    # -------------------------
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/tracks_stub.pkl'
    )

    # Ensure all track lists are aligned with the video length when loading older/partial stubs.
    total_frames = len(video_frames)
    for object_name in ('Players', 'Referees', 'Ball'):
        object_tracks = tracks.get(object_name, [])

        if len(object_tracks) < total_frames:
            object_tracks = object_tracks + [{} for _ in range(total_frames - len(object_tracks))]
        elif len(object_tracks) > total_frames:
            object_tracks = object_tracks[:total_frames]

        tracks[object_name] = object_tracks

    # REQUIRED: adds object positions for later modules
    tracker.add_position_tracks(tracks)

    # -------------------------
    # Camera movement
    # -------------------------
    camera_movement_estimator = CameraMovement(video_frames[0])

    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement.pkl'
    )

    camera_movement_estimator.adjust_positions_tracks(
        tracks,
        camera_movement_per_frame
    )

    # -------------------------
    # View Transformer (FIELD)
    # -------------------------
    view_transformer = ViewTransformer(video_frames[0])

    view_transformer.add_transformed_position_tracks(
        tracks,
        read_from_stub=True,
        stub_path="stubs/field_tracks.pkl"
    )

    # -------------------------
    # Interpolate ball
    # -------------------------
    tracks["Ball"] = tracker.interpolate_ball(tracks["Ball"])

    # -------------------------
    # Speed & Distance
    # -------------------------
    speeddistance_estimator = SpeedDistance()
    speeddistance_estimator.add_speeddistance_tracks(tracks)

    # -------------------------
    # Assign team colors
    # -------------------------
    team_assigner = TeamAssigner()

    # Fallback colors so downstream drawing works even if clustering cannot initialize.
    team_assigner.team_colors[1] = (0, 0, 255)
    team_assigner.team_colors[2] = (255, 0, 0)

    bootstrap_frame_num = None
    bootstrap_players = None

    # Initialize team colors from the first frame with at least 2 valid player detections.
    for frame_num, player_track in enumerate(tracks['Players']):
        if len(player_track) < 2:
            continue

        frame_h, frame_w = video_frames[frame_num].shape[:2]
        valid_players = {}

        for player_id, track in player_track.items():
            bbox = track.get('bbox', None)
            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            x1 = int(max(0, min(frame_w - 1, x1)))
            y1 = int(max(0, min(frame_h - 1, y1)))
            x2 = int(max(0, min(frame_w, x2)))
            y2 = int(max(0, min(frame_h, y2)))

            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            valid_players[player_id] = {'bbox': [x1, y1, x2, y2]}

        if len(valid_players) >= 2:
            bootstrap_frame_num = frame_num
            bootstrap_players = valid_players
            break

    if bootstrap_frame_num is not None:
        team_assigner.assign_team_color(
            video_frames[bootstrap_frame_num],
            bootstrap_players
        )
    else:
        print("Warning: Could not initialize team colors (no frame with >=2 valid players). Using fallback colors.")

    for frame_num, player_track in enumerate(tracks['Players']):

        for player_id, track in player_track.items():

            if hasattr(team_assigner, 'kmeans'):
                team = team_assigner.get_player_team(
                    video_frames[frame_num],
                    track['bbox'],
                    player_id
                )
            else:
                team = 1

            tracks['Players'][frame_num][player_id]['team'] = team
            tracks['Players'][frame_num][player_id]['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))

    # -------------------------
    # Ball possession
    # -------------------------
    player_assigner = PlayerBallAssigner()

    team_ball_control = []
    last_ball_bbox = None

    for frame_num, player_track in enumerate(tracks['Players']):

        ball_dict = tracks['Ball'][frame_num]

        if 1 in ball_dict:
            last_ball_bbox = ball_dict[1]['bbox']

        if last_ball_bbox is None:
            continue

        assigned_player = player_assigner.assign_to_player(
            player_track,
            last_ball_bbox
        )

        if assigned_player != -1:

            tracks['Players'][frame_num][assigned_player]['has_ball'] = True

            team_ball_control.append(
                tracks['Players'][frame_num][assigned_player]['team']
            )

        else:

            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # -------------------------
    # Draw annotations
    # -------------------------
    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        team_ball_control
    )

    # -------------------------
    # Draw camera movement
    # -------------------------
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames,
        camera_movement_per_frame
    )

    # -------------------------
    # Draw speed & distance
    # -------------------------
    speeddistance_estimator.draw_speeddistance(
        output_video_frames,
        tracks
    )

    # -------------------------
    # Draw team structure & formations
    # -------------------------
    team_structure_drawer = TeamStructureDrawer(k_neighbors=3)
    formation_detector = FormationDetector(
        history_size=50,
        distance_threshold=9.0,
        formations_csv_path='Formations.csv'
    )

    for frame_num in range(len(output_video_frames)):

        output_video_frames[frame_num] = team_structure_drawer.draw_team_structure(
            output_video_frames[frame_num],
            tracks['Players'],
            frame_num
        )

        current_formations = formation_detector.update(tracks, frame_num)

        output_video_frames[frame_num] = formation_detector.draw_overlay(
            output_video_frames[frame_num],
            current_formations
        )

    # -------------------------
    # Save video
    # -------------------------
    save_video(
        output_video_frames,
        'output_videos/output_video.avi'
    )


if __name__ == '__main__':
    main()