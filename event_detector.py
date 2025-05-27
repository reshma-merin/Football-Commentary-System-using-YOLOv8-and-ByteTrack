import numpy as np
import cv2
from utils import measure_distance
from utils import get_center_of_bbox

class EventDetector:
    def __init__(self, possession_radius=200, duel_radius=150, speed_threshold=0.5):
        self.possession_radius = possession_radius
        self.duel_radius = duel_radius
        self.speed_threshold = speed_threshold

    def determine_possession(self, players, ball_position):
        close_players = []

        for player_id, player_info in players.items():
            player_pos = player_info.get('position_transformed') or player_info.get('position_adjusted') or player_info.get('position')

            if player_pos is None or ball_position is None:
                continue
            distance = measure_distance(player_pos, ball_position)
            if distance < self.possession_radius:
                close_players.append((player_id, distance))

        if len(close_players) >= 2:
            return "duel", None
        elif len(close_players) == 1:
            return "possession", close_players[0][0]
        else:
            return "no possession", None

    def detect_event(self, previous, current, previous_player, current_player):
        if previous == "possession" and current == "possession":
            if previous_player != current_player:
                return "pass"
        elif previous == "possession" and current == "no possession":
            return "lost possession"
        elif previous == "possession" and current == "duel":
            return "contested"
        return None

    def track_events(self, tracks):
        events = []
        previous_state = ("no possession", None)

        for frame_idx in range(len(tracks['players'])):
            players = tracks['players'][frame_idx]
            ball_info = tracks['ball'][frame_idx].get(1)
            if ball_info is None:
                continue

            ball_position = (
                ball_info.get('position_transformed') or
                ball_info.get('position_adjusted') or
                ball_info.get('position')
            )

            if ball_position is None:
                # Manually compute ball center from bbox
                ball_bbox = ball_info.get('bbox')
                if ball_bbox:
                    ball_position = get_center_of_bbox(ball_bbox)

            control_type, player_id = self.determine_possession(players, ball_position)

            event = self.detect_event(previous_state[0], control_type, previous_state[1], player_id)

            if event:
                events.append({
                    'frame': frame_idx,
                    'event': event,
                    'player_id': player_id
                })

            previous_state = (control_type, player_id)

        return events

    def draw_events(self, frames, events):
        output_frames = []
        event_dict = {event['frame']: event for event in events}

        for frame_idx, frame in enumerate(frames):
            frame = frame.copy()

            # Instead of exact frame match, draw if near
            for event_frame in event_dict.keys():
                if abs(frame_idx - event_frame) <= 3:  # Allow small window
                    event = event_dict[event_frame]
                    text = f"{event['event'].upper()}"  # Make it uppercase
                    cv2.rectangle(frame, (30, 50), (450, 130), (255, 255, 255), -1)  # White background box
                    cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)  # Black text
                    break

            output_frames.append(frame)


        return output_frames


