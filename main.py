from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from event_detector import EventDetector
from commentary_generator import CommentaryGenerator
from text_to_speech import TextToSpeech
import os
import json

def main():
    # Read Video
    video_frames = read_video(r"C:\Users\reshmamerinthomas\Desktop\football_analysis\input_videos\08fd33_4.mp4")

    # Initialize Tracker
    tracker = Tracker(r'C:\Users\reshmamerinthomas\Desktop\football_analysis\models\best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
    
    def convert_tracking_to_event_format(tracking_results):
        """
        Convert tracking results to the format expected by the event detector
        
        Args:
            tracking_results: Your tracking system's output format
            
        Returns:
            Dictionary in the format expected by FootballEventDetector
        """
        # Initialize the event format structure
        event_format = {
            'ball': {},
            'players': {}
        }
        
        # Process ball tracking data
        for frame_id, frame_data in tracking_results.items():
            frame_id = int(frame_id)  # Ensure frame_id is an integer
            
            # Adapt this to match your tracking output structure
            if 'ball' in frame_data and frame_data['ball'] is not None:
                ball_pos = frame_data['ball'].get('position', None)
                if ball_pos is not None:
                    event_format['ball'][frame_id] = {
                        'position': ball_pos
                    }
    
    
    # Process player tracking data
        for frame_id, frame_data in tracking_results.items():
            frame_id = int(frame_id)
            
            # Adapt this to match your tracking output structure
            if 'players' in frame_data:
                for player_id, player_data in frame_data['players'].items():
                    if player_id not in event_format['players']:
                        event_format['players'][player_id] = {}
                    
                    if player_data is not None and 'position' in player_data:
                        # Add team information if available
                        team_id = player_data.get('team_id', None)
                        
                        event_format['players'][player_id][frame_id] = {
                            'position': player_data['position'],
                            'team': team_id
                        }
        
        return event_format

    def run_event_detection(tracking_results, output_video_path):
        """
        Run event detection on tracking results
        
        Args:
            tracking_results: Dictionary containing tracking data
            output_video_path: Path to the output video
            
        Returns:
            Detected events
        """
        # Convert tracking results to the format expected by the event detector
        event_format_data = convert_tracking_to_event_format(tracking_results)
        
        # Configure the event detector
        config = {
            "possession_zone_radius": 1.0,
            "duel_zone_radius": 1.0,
            "pitch_length": 105.0,  # Standard football pitch length in meters
            "pitch_width": 68.0,    # Standard football pitch width in meters
            "frame_rate": 25        # Adjust based on your video
        }
        
        # Initialize and run the event detector
        print("Detecting events...")
        event_detector = EventDetector(config)
        events = event_detector.detect_events(event_format_data)
        print(f"Number of detected events: {len(events)}")
        print(events[:5])  # show first few events



        print(f"Detected {len(events)} events")
        
        # Generate event statistics
        stats = event_detector.generate_event_stats()
        print("\nEvent Statistics:")
        print(f"Total events: {stats.get('total_events', 0)}")
        
        print("\nEvent Types:")
        for event_type, count in stats.get('event_types', {}).items():
            print(f"  {event_type}: {count}")
        
        # Save the events
        output_dir = os.path.dirname(output_video_path)
        if not output_dir:
            output_dir = '.'
            
        # Create event results directory
        event_dir = os.path.join(output_dir, "event_detection")
        os.makedirs(event_dir, exist_ok=True)
        
        # Export events to files
        json_path = os.path.join(event_dir, "events.json")
        csv_path = os.path.join(event_dir, "events.csv")
        
        event_detector.export_events_to_json(json_path)
        event_detector.export_events_to_csv(csv_path)
        
        # Save statistics
        stats_path = os.path.join(event_dir, "event_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Visualize events on the video (optional)
        print("\nGenerating video with event visualization...")
        events_video_path = os.path.join(event_dir, "events_visualization.mp4")
        event_detector.visualize_events(output_video_path, events_video_path)
        print(f"Event visualization saved to {events_video_path}")
        
        return events






    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    event_detector = EventDetector(possession_radius = 200)
    print(tracks['players'][0])
    print(tracks['ball'][0])

    events = event_detector.track_events(tracks)

    print(f"Number of events detected: {len(events)}")  # Debugging
    output_video_frames = event_detector.draw_events(output_video_frames, events)

    commentary_gen = CommentaryGenerator()
    tts_engine = TextToSpeech()

    all_commentaries = []

    # Generate commentary for all detected events
    for event in events:
        commentary = commentary_gen.generate_commentary(event['event'], event.get('player_id'))
        if commentary:
            all_commentaries.append(commentary)  # for now, print to console


    full_commentary_text = " ".join(all_commentaries)

     # Create folder if not exists
    if not os.path.exists('audio_commentaries'):
        os.makedirs('audio_commentaries')

    output_audio_path = r"audio_commentaries/full_match_commentary.mp3"
    tts_engine.text_to_audio(full_commentary_text, output_audio_path)

    save_video(output_video_frames, 'output_videos/output_video_1.avi')

    output_video_file = r"C:\Users\reshmamerinthomas\Desktop\football_analysis\output_videos\output_video_1.avi"
    
    output_video_frames = event_detector.draw_events(output_video_frames, events)






if __name__ == '__main__':
    main()