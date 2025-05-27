import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import cv2

class FootballEventDetector:
    """
    Football event detection system based on tracking data,
    implementing the approach from Vidal-Codina et al. (2022).
    """
    
    def __init__(self, config=None):
        """
        Initialize the event detector with configuration parameters
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        self.config = {
            # Possession zone radius (meters) - distance within which player has possession
            "possession_zone_radius": 1.0,
            
            # Duel zone radius (meters) - distance within which multiple players are in a duel
            "duel_zone_radius": 1.0,
            
            # Minimum ball displacement to detect a possession loss (meters)
            "min_ball_displacement": 0.3,
            
            # Minimum change in ball direction to detect possession gain (radians)
            "min_direction_change": 0.3,
            
            # Minimum change in ball speed to detect possession gain (m/s)
            "min_speed_change": 1.0,
            
            # Pitch dimensions (meters)
            "pitch_length": 105.0,
            "pitch_width": 68.0,
            
            # Zones for shot vs cross detection
            "shot_zone_width": 25.0,  # Central width for shot zone
            "cross_zone_width": 30.0,  # Width from sideline for cross zone
            
            # Set piece detection tolerances
            "kickoff_tolerance": 2.0,       # Distance from center mark
            "corner_tolerance": 2.0,        # Distance from corner mark
            "goal_area_tolerance": 2.0,     # Distance from goal area
            "throw_in_tolerance": 2.0,      # Distance from sideline
            "penalty_mark_tolerance": 2.0,  # Distance from penalty mark
            
            # Frame rate of tracking data (Hz)
            "frame_rate": 25
        }
        
        # Update with user-provided config
        if config:
            self.config.update(config)
            
        # Initialize storage for detected events
        self.events = []
    
    def detect_events(self, tracking_data: Dict) -> List[Dict]:
        """
        Main method to detect events from tracking data
        
        Args:
            tracking_data: Dictionary containing ball and player tracking data
                Format: {
                    'ball': {frame_id: {'position': [x, y]}},
                    'players': {player_id: {frame_id: {'position': [x, y], 'team': team_id}}}
                }
                
        Returns:
            List of detected events
        """
        # Step 1: Determine ball possession
        possession_data = self._determine_possession(tracking_data)
        
        # Step 2: Detect events from possession changes
        events = self._detect_events_from_possession(possession_data, tracking_data)
        
        self.events = events
        return events
    
    def _determine_possession(self, tracking_data: Dict) -> Dict:
        """
        Determine ball possession for each frame
        
        Args:
            tracking_data: Dictionary with player and ball tracking data
            
        Returns:
            Dictionary with possession information
        """
        # Initialize storage for possession data
        possession_data = {
            'ball_control': {},  # 'possession', 'duel', 'no_possession', or 'dead_ball'
            'possession_player': {},  # player_id if in possession
            'possession_team': {},  # team_id if in possession
            'possession_gains': [],  # frames where possession is gained
            'possession_losses': [],  # frames where possession is lost
            'set_piece_triggers': {}  # frames with set piece triggers
        }
        
        ball_data = tracking_data['ball']
        players_data = tracking_data['players']
        
        # Determine possession frames
        all_frames = sorted(list(ball_data.keys()))
        
        for frame_id in all_frames:
            # Skip if ball data is missing (dead ball)
            if frame_id not in ball_data or 'position' not in ball_data[frame_id]:
                possession_data['ball_control'][frame_id] = 'dead_ball'
                continue
            
            ball_pos = ball_data[frame_id]['position']
            
            # Find players close to the ball
            players_near_ball = []
            for player_id, player_frames in players_data.items():
                if frame_id in player_frames and 'position' in player_frames[frame_id]:
                    player_pos = player_frames[frame_id]['position']
                    distance = np.linalg.norm(np.array(ball_pos) - np.array(player_pos))
                    
                    if distance <= self.config['possession_zone_radius']:
                        players_near_ball.append({
                            'player_id': player_id,
                            'distance': distance,
                            'team': player_frames[frame_id].get('team', None)
                        })
            
            # Sort players by distance to ball
            players_near_ball.sort(key=lambda x: x['distance'])
            
            # Check for possession or duel
            if len(players_near_ball) == 0:
                possession_data['ball_control'][frame_id] = 'no_possession'
            elif len(players_near_ball) == 1:
                possession_data['ball_control'][frame_id] = 'possession'
                possession_data['possession_player'][frame_id] = players_near_ball[0]['player_id']
                possession_data['possession_team'][frame_id] = players_near_ball[0]['team']
            else:
                # Check if players are from different teams (duel)
                teams = set(player['team'] for player in players_near_ball if player['team'])
                if len(teams) >= 2:
                    possession_data['ball_control'][frame_id] = 'duel'
                else:
                    # Same team, closest player has possession
                    possession_data['ball_control'][frame_id] = 'possession'
                    possession_data['possession_player'][frame_id] = players_near_ball[0]['player_id']
                    possession_data['possession_team'][frame_id] = players_near_ball[0]['team']
        
        # Detect possession gains and losses
        for i in range(1, len(all_frames)):
            prev_frame = all_frames[i-1]
            curr_frame = all_frames[i]
            
            # Detect possession gains
            if (possession_data['ball_control'][curr_frame] == 'possession' and 
                (possession_data['ball_control'][prev_frame] != 'possession' or 
                 possession_data['possession_player'][prev_frame] != possession_data['possession_player'][curr_frame])):
                
                # Check if ball direction or speed changed using surrounding frames
                ball_changed = self._check_ball_change(tracking_data, curr_frame)
                
                if ball_changed:
                    possession_data['possession_gains'].append({
                        'frame': curr_frame,
                        'player_id': possession_data['possession_player'][curr_frame],
                        'team': possession_data['possession_team'][curr_frame]
                    })
            
            # Detect possession losses
            if (possession_data['ball_control'][prev_frame] == 'possession' and 
                (possession_data['ball_control'][curr_frame] != 'possession' or 
                 possession_data['possession_player'][prev_frame] != possession_data['possession_player'][curr_frame])):
                
                # Check ball displacement
                if prev_frame in ball_data and curr_frame in ball_data:
                    prev_pos = ball_data[prev_frame]['position']
                    curr_pos = ball_data[curr_frame]['position']
                    displacement = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
                    
                    if displacement >= self.config['min_ball_displacement']:
                        possession_data['possession_losses'].append({
                            'frame': prev_frame,
                            'player_id': possession_data['possession_player'][prev_frame],
                            'team': possession_data['possession_team'][prev_frame]
                        })
        
        # Detect set piece triggers
        dead_ball_intervals = self._detect_dead_ball_intervals(possession_data['ball_control'])
        
        for interval in dead_ball_intervals:
            start_frame, end_frame = interval
            set_piece_triggers = self._detect_set_piece_triggers(tracking_data, start_frame, end_frame)
            
            if set_piece_triggers:
                possession_data['set_piece_triggers'][interval] = set_piece_triggers
        
        return possession_data
    
    def _check_ball_change(self, tracking_data: Dict, frame_id: int) -> bool:
        """
        Check if ball direction or speed changed around the given frame
        
        Args:
            tracking_data: Dictionary with ball tracking data
            frame_id: Frame to check
            
        Returns:
            True if ball changed direction or speed
        """
        ball_data = tracking_data['ball']
        frames = sorted(list(ball_data.keys()))
        
        # Find position of frame_id in frames list
        try:
            idx = frames.index(frame_id)
        except ValueError:
            return False
        
        # Need at least 3 frames to check direction change
        if idx < 2 or idx >= len(frames) - 2:
            return False
        
        # Get positions from surrounding frames
        pos_prev2 = ball_data[frames[idx-2]].get('position')
        pos_prev1 = ball_data[frames[idx-1]].get('position')
        pos_curr = ball_data[frames[idx]].get('position')
        pos_next1 = ball_data[frames[idx+1]].get('position')
        
        if None in [pos_prev2, pos_prev1, pos_curr, pos_next1]:
            return False
        
        # Calculate velocities
        vel_prev = np.array(pos_prev1) - np.array(pos_prev2)
        vel_curr = np.array(pos_curr) - np.array(pos_prev1)
        vel_next = np.array(pos_next1) - np.array(pos_curr)
        
        # Calculate speeds
        speed_prev = np.linalg.norm(vel_prev)
        speed_curr = np.linalg.norm(vel_curr)
        speed_next = np.linalg.norm(vel_next)
        
        # Check speed change
        speed_change = max(abs(speed_curr - speed_prev), abs(speed_next - speed_curr))
        if speed_change >= self.config['min_speed_change']:
            return True
        
        # Check direction change (dot product of normalized velocities)
        if speed_prev > 0 and speed_curr > 0:
            vel_prev_norm = vel_prev / speed_prev
            vel_curr_norm = vel_curr / speed_curr
            dot_product = np.dot(vel_prev_norm, vel_curr_norm)
            angle_change = np.arccos(max(min(dot_product, 1.0), -1.0))
            
            if angle_change >= self.config['min_direction_change']:
                return True
        
        if speed_curr > 0 and speed_next > 0:
            vel_curr_norm = vel_curr / speed_curr
            vel_next_norm = vel_next / speed_next
            dot_product = np.dot(vel_curr_norm, vel_next_norm)
            angle_change = np.arccos(max(min(dot_product, 1.0), -1.0))
            
            if angle_change >= self.config['min_direction_change']:
                return True
        
        return False
    
    def _detect_dead_ball_intervals(self, ball_control: Dict) -> List[Tuple[int, int]]:
        """
        Detect intervals where the ball is dead
        
        Args:
            ball_control: Dictionary with ball control information
            
        Returns:
            List of (start_frame, end_frame) intervals
        """
        frames = sorted(list(ball_control.keys()))
        intervals = []
        
        start_frame = None
        for frame in frames:
            if ball_control[frame] == 'dead_ball':
                if start_frame is None:
                    start_frame = frame
            else:
                if start_frame is not None:
                    intervals.append((start_frame, frame - 1))
                    start_frame = None
        
        # Handle last interval
        if start_frame is not None:
            intervals.append((start_frame, frames[-1]))
        
        return intervals
    
    def _detect_set_piece_triggers(self, tracking_data: Dict, start_frame: int, end_frame: int) -> Dict:
        """
        Detect set piece triggers in a dead ball interval
        
        Args:
            tracking_data: Dictionary with player and ball tracking data
            start_frame: Start frame of dead ball interval
            end_frame: End frame of dead ball interval
            
        Returns:
            Dictionary with triggered set pieces
        """
        # Use middle frame of interval for detection
        frame_to_check = (start_frame + end_frame) // 2
        
        # Skip if no player data for this frame
        if frame_to_check not in list(tracking_data['players'].values())[0]:
            return {}
        
        # Get all player positions for this frame
        player_positions = {}
        player_teams = {}
        
        for player_id, player_data in tracking_data['players'].items():
            if frame_to_check in player_data and 'position' in player_data[frame_to_check]:
                player_positions[player_id] = player_data[frame_to_check]['position']
                if 'team' in player_data[frame_to_check]:
                    player_teams[player_id] = player_data[frame_to_check]['team']
        
        triggers = {}
        
        # Check kickoff trigger
        if self._check_kickoff_trigger(player_positions, player_teams):
            triggers['kickoff'] = True
        
        # Check corner kick trigger
        corner_result = self._check_corner_trigger(player_positions, player_teams)
        if corner_result:
            triggers['corner_kick'] = corner_result
        
        # Check throw-in trigger
        throw_in_result = self._check_throw_in_trigger(player_positions, player_teams)
        if throw_in_result:
            triggers['throw_in'] = throw_in_result
        
        # Check goal kick trigger
        goal_kick_result = self._check_goal_kick_trigger(player_positions, player_teams)
        if goal_kick_result:
            triggers['goal_kick'] = goal_kick_result
        
        # Check penalty kick trigger
        penalty_result = self._check_penalty_trigger(player_positions, player_teams)
        if penalty_result:
            triggers['penalty_kick'] = penalty_result
        
        return triggers
    
    def _check_kickoff_trigger(self, player_positions, player_teams):
        """Check if players are in kickoff configuration"""
        # Center of pitch
        center = [self.config['pitch_length'] / 2, self.config['pitch_width'] / 2]
        
        # Check if any player is near center
        player_at_center = False
        for player_id, position in player_positions.items():
            dist_to_center = np.linalg.norm(np.array(position) - np.array(center))
            if dist_to_center <= self.config['kickoff_tolerance']:
                player_at_center = True
                break
        
        if not player_at_center:
            return False
        
        # Check if all players are in their own half (with tolerance)
        if not player_teams:
            return False  # No team info available
        
        team_ids = set(player_teams.values())
        if len(team_ids) < 2:
            return False  # Need at least two teams
        
        # Divide players by team
        players_by_team = {team_id: [] for team_id in team_ids}
        for player_id, team_id in player_teams.items():
            if player_id in player_positions:
                players_by_team[team_id].append(player_positions[player_id])
        
        # Check if teams are on opposite sides of halfway line
        halfway_line = self.config['pitch_length'] / 2
        tolerance = self.config['kickoff_tolerance']
        
        teams_on_opposite_sides = True
        team_sides = {}
        
        for team_id, positions in players_by_team.items():
            if not positions:
                continue
                
            # Check which side majority of players are on
            positions_array = np.array(positions)
            x_positions = positions_array[:, 0]
            
            left_side_count = np.sum(x_positions < halfway_line - tolerance)
            right_side_count = np.sum(x_positions > halfway_line + tolerance)
            
            if left_side_count > right_side_count:
                team_sides[team_id] = 'left'
            else:
                team_sides[team_id] = 'right'
        
        # Ensure teams are on different sides
        if len(set(team_sides.values())) < 2:
            teams_on_opposite_sides = False
        
        return teams_on_opposite_sides and player_at_center
    
    def _check_corner_trigger(self, player_positions, player_teams):
        """Check if any player is at corner position"""
        # Define corner positions
        corners = [
            [0, 0],  # Bottom left
            [0, self.config['pitch_width']],  # Top left
            [self.config['pitch_length'], 0],  # Bottom right
            [self.config['pitch_length'], self.config['pitch_width']]  # Top right
        ]
        
        for player_id, position in player_positions.items():
            for corner in corners:
                dist_to_corner = np.linalg.norm(np.array(position) - np.array(corner))
                if dist_to_corner <= self.config['corner_tolerance']:
                    # Determine which corner
                    if corner[0] < self.config['pitch_length'] / 2:
                        side = 'left'
                    else:
                        side = 'right'
                        
                    if corner[1] < self.config['pitch_width'] / 2:
                        vertical = 'bottom'
                    else:
                        vertical = 'top'
                    
                    return {'player_id': player_id, 'corner': f"{vertical}_{side}"}
        
        return None
    
    def _check_throw_in_trigger(self, player_positions, player_teams):
        """Check if any player is at sideline position"""
        for player_id, position in player_positions.items():
            x, y = position
            
            # Check if player is near sideline
            near_bottom = y <= self.config['throw_in_tolerance']
            near_top = y >= (self.config['pitch_width'] - self.config['throw_in_tolerance'])
            
            if near_bottom or near_top:
                # Determine which sideline
                if near_bottom:
                    side = 'bottom'
                else:
                    side = 'top'
                
                # Determine approximate x position
                x_rel = x / self.config['pitch_length']
                x_pos = 'middle'
                
                if x_rel < 0.33:
                    x_pos = 'defensive'
                elif x_rel > 0.66:
                    x_pos = 'attacking'
                
                return {'player_id': player_id, 'side': side, 'position': x_pos}
        
        return None
    
    def _check_goal_kick_trigger(self, player_positions, player_teams):
        """Check if any player is in goal area position"""
        # Define goal areas
        goal_area_left = [
            [0, (self.config['pitch_width'] - 18.32) / 2],
            [5.5, (self.config['pitch_width'] + 18.32) / 2]
        ]
        
        goal_area_right = [
            [self.config['pitch_length'] - 5.5, (self.config['pitch_width'] - 18.32) / 2],
            [self.config['pitch_length'], (self.config['pitch_width'] + 18.32) / 2]
        ]
        
        for player_id, position in player_positions.items():
            x, y = position
            
            # Check left goal area
            if (goal_area_left[0][0] - self.config['goal_area_tolerance'] <= x <= goal_area_left[1][0] + self.config['goal_area_tolerance'] and
                goal_area_left[0][1] - self.config['goal_area_tolerance'] <= y <= goal_area_left[1][1] + self.config['goal_area_tolerance']):
                return {'player_id': player_id, 'side': 'left'}
            
            # Check right goal area
            if (goal_area_right[0][0] - self.config['goal_area_tolerance'] <= x <= goal_area_right[1][0] + self.config['goal_area_tolerance'] and
                goal_area_right[0][1] - self.config['goal_area_tolerance'] <= y <= goal_area_right[1][1] + self.config['goal_area_tolerance']):
                return {'player_id': player_id, 'side': 'right'}
        
        return None
    
    def _check_penalty_trigger(self, player_positions, player_teams):
        """Check if a player is at penalty spot"""
        # Define penalty spots
        penalty_left = [11.0, self.config['pitch_width'] / 2]
        penalty_right = [self.config['pitch_length'] - 11.0, self.config['pitch_width'] / 2]
        
        for player_id, position in player_positions.items():
            # Check left penalty spot
            dist_to_left = np.linalg.norm(np.array(position) - np.array(penalty_left))
            if dist_to_left <= self.config['penalty_mark_tolerance']:
                return {'player_id': player_id, 'side': 'left'}
            
            # Check right penalty spot
            dist_to_right = np.linalg.norm(np.array(position) - np.array(penalty_right))
            if dist_to_right <= self.config['penalty_mark_tolerance']:
                return {'player_id': player_id, 'side': 'right'}
        
        return None
    
    def _detect_events_from_possession(self, possession_data: Dict, tracking_data: Dict) -> List[Dict]:
        """
        Detect football events from possession changes
        
        Args:
            possession_data: Dictionary with possession information
            tracking_data: Dictionary with player and ball tracking data
            
        Returns:
            List of detected events
        """
        events = []
        
        # Process possession gains and losses
        for gain in possession_data['possession_gains']:
            frame = gain['frame']
            player_id = gain['player_id']
            team = gain['team']
            
            # Look for preceding possession loss
            preceding_loss = None
            for loss in possession_data['possession_losses']:
                if loss['frame'] < frame and (preceding_loss is None or loss['frame'] > preceding_loss['frame']):
                    preceding_loss = loss
            
            if preceding_loss:
                # Determine event type based on players involved
                source_player_id = preceding_loss['player_id']
                source_team = preceding_loss['team']
                
                # Get positions for analysis
                ball_pos_at_loss = None
                if preceding_loss['frame'] in tracking_data['ball']:
                    ball_pos_at_loss = tracking_data['ball'][preceding_loss['frame']]['position']
                
                ball_pos_at_gain = None
                if frame in tracking_data['ball']:
                    ball_pos_at_gain = tracking_data['ball'][frame]['position']
                
                # If positions available, classify event
                if ball_pos_at_loss and ball_pos_at_gain:
                    # Same team -> pass
                    if team == source_team:
                        event_type = self._classify_pass_type(ball_pos_at_loss, ball_pos_at_gain, tracking_data, frame)
                        
                        events.append({
                            'frame': preceding_loss['frame'],
                            'event_type': event_type,
                            'sub_type': None,
                            'source_player_id': source_player_id,
                            'target_player_id': player_id,
                            'source_team': source_team,
                            'target_team': team,
                            'start_position': ball_pos_at_loss,
                            'end_position': ball_pos_at_gain
                        })
                    # Different team -> interception
                    else:
                        events.append({
                            'frame': frame,
                            'event_type': 'interception',
                            'sub_type': None,
                            'source_player_id': source_player_id,
                            'target_player_id': player_id,
                            'source_team': source_team,
                            'target_team': team,
                            'start_position': ball_pos_at_loss,
                            'end_position': ball_pos_at_gain
                        })
            else:
                # Possession gain without preceding loss
                # This could be a reception from loose ball
                if frame in tracking_data['ball']:
                    ball_pos = tracking_data['ball'][frame]['position']
                    
                    events.append({
                        'frame': frame,
                        'event_type': 'reception_loose_ball',
                        'sub_type': None,
                        'source_player_id': None,
                        'target_player_id': player_id,
                        'source_team': None,
                        'target_team': team,
                        'start_position': None,
                        'end_position': ball_pos
                    })
        
        # Process dead ball intervals and set pieces
        dead_ball_intervals = self._detect_dead_ball_intervals(possession_data['ball_control'])
        
        for interval in dead_ball_intervals:
            start_frame, end_frame = interval
            
            # Check if there's a set piece trigger for this interval
            if interval in possession_data['set_piece_triggers']:
                triggers = possession_data['set_piece_triggers'][interval]
                
                # Determine dead ball event (what happened before interval)
                dead_ball_event = self._determine_dead_ball_event(start_frame, triggers, tracking_data, events)
                
                if dead_ball_event:
                    events.append(dead_ball_event)
                
                # Determine set piece event (what happened after interval)
                set_piece_event = self._determine_set_piece_event(end_frame, triggers, tracking_data)
                
                if set_piece_event:
                    events.append(set_piece_event)
        
        # Sort events by frame
        events.sort(key=lambda x: x['frame'])
        
        return events
    
    def _classify_pass_type(self, start_pos, end_pos, tracking_data, frame):
        """Classify pass into regular pass, cross, or shot"""
        x_start, y_start = start_pos
        x_end, y_end = end_pos
        
        # Convert to normalized coordinates (0-100)
        x_start_norm = 100 * x_start / self.config['pitch_length']
        x_end_norm = 100 * x_end / self.config['pitch_length']
        
        # Check shot - ball moving toward goal and in final third
        if x_start_norm > 66 and x_end_norm > x_start_norm:
            # Check if in shot zone (central area)
            y_center = self.config['pitch_width'] / 2
            y_distance_from_center = abs(y_start - y_center)
            
            if y_distance_from_center < self.config['shot_zone_width'] / 2:
                # Check if any opponent in goal area (goalkeeper)
                goalkeeper_present = False
                attacking_team = None
                
                # Find team of passer
                for player_id, player_data in tracking_data['players'].items():
                    if frame in player_data and player_data[frame].get('position') == start_pos:
                        attacking_team = player_data[frame].get('team')
                        break
                
                if attacking_team:
                    # Check for goalkeeper
                    goal_line = self.config['pitch_length']
                    goal_center_y = self.config['pitch_width'] / 2
                    
                    for player_id, player_data in tracking_data['players'].items():
                        if frame not in player_data:
                            continue
                            
                        if player_data[frame].get('team') != attacking_team:
                            player_pos = player_data[frame].get('position')
                            if player_pos:
                                # Check if player is near goal line and center
                                if (player_pos[0] > goal_line - 3 and 
                                    abs(player_pos[1] - goal_center_y) < 5):
                                    goalkeeper_present = True
                                    break
                
                # Classify shot based on target
                return 'shot'
            
        # Check cross - from wide area to central area
        y_center = self.config['pitch_width'] / 2
        cross_zone_width = self.config['cross_zone_width']
        
        # Starting from wide area
        if (y_start < cross_zone_width or y_start > self.config['pitch_width'] - cross_zone_width):
            # Ending in central area
            if abs(y_end - y_center) < self.config['pitch_width'] / 4:
                return 'cross'
        
        # Default to regular pass
        return 'pass'
    
    def _determine_dead_ball_event(self, start_frame, triggers, tracking_data, previous_events):
        """Determine what event caused the dead ball interval"""
        # Try to find last event before dead ball
        last_event_frame = 0
        last_event_type = None
        
        for event in previous_events:
            if event['frame'] < start_frame and event['frame'] > last_event_frame:
                last_event_frame = event['frame']
                last_event_type = event['event_type']
        
        # If last event was shot, check for goal
        if last_event_type == 'shot' and 'kickoff' in triggers:
            return {
                'frame': start_frame - 1,
                'event_type': 'goal',
                'sub_type': None,
                'source_player_id': None,  # Could find this from previous events
                'target_player_id': None,
                'source_team': None,
                'target_team': None,
                'start_position': None,
                'end_position': None
            }
        
        # Determine based on triggers
        if 'corner_kick' in triggers:
            return {
                'frame': start_frame,
                'event_type': 'out_for_corner_kick',
                'sub_type': triggers}
        
    def _determine_set_piece_event(self, end_frame, triggers, tracking_data):
        """Determine what set piece resumes play after dead ball interval"""
        # Find first possession after dead ball
        next_frame = end_frame + 1
        
        # Get ball position after dead ball
        ball_pos = None
        if next_frame in tracking_data['ball']:
            ball_pos = tracking_data['ball'][next_frame]['position']
        else:
            # Try to find next available ball position
            for frame in range(next_frame, next_frame + 10):
                if frame in tracking_data['ball'] and 'position' in tracking_data['ball'][frame]:
                    ball_pos = tracking_data['ball'][frame]['position']
                    next_frame = frame
                    break
        
        if not ball_pos:
            return None
        
        # Find player closest to ball at restart
        closest_player_id = None
        closest_player_team = None
        min_distance = float('inf')
        
        for player_id, player_data in tracking_data['players'].items():
            if next_frame not in player_data or 'position' not in player_data[next_frame]:
                continue
                
            player_pos = player_data[next_frame]['position']
            distance = np.linalg.norm(np.array(ball_pos) - np.array(player_pos))
            
            if distance < min_distance:
                min_distance = distance
                closest_player_id = player_id
                closest_player_team = player_data[next_frame].get('team')
        
        # Determine set piece type based on triggers
        event_type = 'free_kick'  # Default
        sub_type = None
        
        if 'kickoff' in triggers:
            event_type = 'kickoff'
        elif 'corner_kick' in triggers:
            event_type = 'corner_kick'
            sub_type = triggers['corner_kick']['corner'] if 'corner' in triggers['corner_kick'] else None
        elif 'throw_in' in triggers:
            event_type = 'throw_in'
            sub_type = f"{triggers['throw_in'].get('side', '')}_{triggers['throw_in'].get('position', '')}" if 'throw_in' in triggers else None
        elif 'goal_kick' in triggers:
            event_type = 'goal_kick'
            sub_type = triggers['goal_kick'].get('side', None) if 'goal_kick' in triggers else None
        elif 'penalty_kick' in triggers:
            event_type = 'penalty_kick'
            sub_type = triggers['penalty_kick'].get('side', None) if 'penalty_kick' in triggers else None
        
        return {
            'frame': next_frame,
            'event_type': event_type,
            'sub_type': sub_type,
            'source_player_id': closest_player_id,
            'target_player_id': None,
            'source_team': closest_player_team,
            'target_team': None,
            'start_position': ball_pos,
            'end_position': None
        }
    
    def visualize_events(self, video_path=None, output_path=None, frame_range=None):
        """
        Visualize detected events on video frames
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (if None, no video is saved)
            frame_range: Range of frames to process (start, end)
            
        Returns:
            None
        """
        if not video_path:
            return
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Set frame range
        if frame_range is None:
            frame_range = (0, frame_count - 1)
        
        # Create event lookup by frame
        events_by_frame = {}
        for event in self.events:
            events_by_frame[event['frame']] = event
        
        # Process frames
        for frame_id in range(frame_range[0], frame_range[1] + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if there's an event for this frame
            if frame_id in events_by_frame:
                event = events_by_frame[frame_id]
                self._draw_event_on_frame(frame, event, width, height)
            
            # Display frame
            if output_path is None:
                cv2.imshow('Frame', frame)
                cv2.waitKey(25)  # Wait for 25ms
            
            # Write frame to output video
            if out:
                out.write(frame)
        
        # Release resources
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    
    def _draw_event_on_frame(self, frame, event, width, height):
        """Draw event information on video frame"""
        event_type = event['event_type']
        
        # Set color based on event type
        color = (255, 255, 255)  # Default: white
        
        if event_type == 'pass':
            color = (0, 255, 0)  # Green
        elif event_type == 'cross':
            color = (0, 255, 255)  # Yellow
        elif event_type == 'shot':
            color = (0, 0, 255)  # Red
        elif event_type == 'interception':
            color = (255, 0, 0)  # Blue
        elif event_type == 'goal':
            color = (0, 0, 255)  # Red
        
        # Draw event label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{event_type.upper()}"
        
        if event['sub_type']:
            text += f" ({event['sub_type']})"
            
        # Add player information if available
        if event['source_player_id']:
            text += f" | Player {event['source_player_id']}"
            
        if event['target_player_id'] and event['target_player_id'] != event['source_player_id']:
            text += f" to {event['target_player_id']}"
        
        # Draw background box
        cv2.rectangle(frame, (10, 10), (width - 10, 80), color, -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 80), (255, 255, 255), 2)
        
        # Draw text
        cv2.putText(frame, text, (30, 50), font, 0.7, (0, 0, 0), 2)
        
        # Draw positions if available
        if event['start_position'] and event['end_position']:
            # Convert from pitch coordinates to frame coordinates
            # This requires knowing how the pitch is mapped to the video frame
            # For simplicity, we'll assume a top-down view with pitch filling the frame
            start_x = int(event['start_position'][0] * width / self.config['pitch_length'])
            start_y = int(event['start_position'][1] * height / self.config['pitch_width'])
            
            end_x = int(event['end_position'][0] * width / self.config['pitch_length'])
            end_y = int(event['end_position'][1] * height / self.config['pitch_width'])
            
            # Draw line for the event
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
            
            # Draw circles at start and end points
            cv2.circle(frame, (start_x, start_y), 5, color, -1)
            cv2.circle(frame, (end_x, end_y), 5, color, -1)
    
def generate_event_stats(self):
        """Generate statistics from detected events"""
        if not self.events:
            return {'error': 'No events detected'}
        
        stats = {
            'total_events': len(self.events),
            'event_types': {},
            'player_stats': {},
            'team_stats': {}
        }
        
        # Count event types
        for event in self.events:
            event_type = event['event_type']
            
            if event_type not in stats['event_types']:
                stats['event_types'][event_type] = 0
            
            stats['event_types'][event_type] += 1
            
            # Player stats
            if event['source_player_id']:
                player_id = event['source_player_id']
                
                if player_id not in stats['player_stats']:
                    stats['player_stats'][player_id] = {
                        'passes': 0,
                        'crosses': 0,
                        'shots': 0,
                        'goals': 0,
                        'interceptions': 0
                    }
                
                if event_type == 'pass':
                    stats['player_stats'][player_id]['passes'] += 1
                elif event_type == 'cross':
                    stats['player_stats'][player_id]['crosses'] += 1
                elif event_type == 'shot':
                    stats['player_stats'][player_id]['shots'] += 1
                elif event_type == 'goal':
                    stats['player_stats'][player_id]['goals'] += 1
                elif event_type == 'interception':
                    stats['player_stats'][player_id]['interceptions'] += 1
            
            # Team stats
            if event['source_team']:
                team_id = event['source_team']
                
                if team_id not in stats['team_stats']:
                    stats['team_stats'][team_id] = {
                        'passes': 0,
                        'crosses': 0,
                        'shots': 0,
                        'goals': 0,
                        'interceptions': 0,
                        'corners': 0,
                        'throw_ins': 0,
                        'free_kicks': 0
                    }
                
                if event_type == 'pass':
                    stats['team_stats'][team_id]['passes'] += 1
                elif event_type == 'cross':
                    stats['team_stats'][team_id]['crosses'] += 1
                elif event_type == 'shot':
                    stats['team_stats'][team_id]['shots'] += 1
                elif event_type == 'goal':
                    stats['team_stats'][team_id]['goals'] += 1
                elif event_type == 'interception':
                    stats['team_stats'][team_id]['interceptions'] += 1
                elif event_type == 'corner_kick':
                    stats['team_stats'][team_id]['corners'] += 1
                elif event_type == 'throw_in':
                    stats['team_stats'][team_id]['throw_ins'] += 1
                elif event_type == 'free_kick':
                    stats['team_stats'][team_id]['free_kicks'] += 1
        
        return stats
    
    def export_events_to_json(self, output_path):
        """Export detected events to JSON file"""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(self.events, f, indent=2)
            
        print(f"Exported {len(self.events)} events to {output_path}")
    
    def export_events_to_csv(self, output_path):
        """Export detected events to CSV file"""
        import csv
        
        fields = [
            'frame', 'event_type', 'sub_type', 'source_player_id', 'target_player_id',
            'source_team', 'target_team', 'start_position_x', 'start_position_y',
            'end_position_x', 'end_position_y'
        ]
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            
            for event in self.events:
                row = {
                    'frame': event['frame'],
                    'event_type': event['event_type'],
                    'sub_type': event['sub_type'] or '',
                    'source_player_id': event['source_player_id'] or '',
                    'target_player_id': event['target_player_id'] or '',
                    'source_team': event['source_team'] or '',
                    'target_team': event['target_team'] or '',
                    'start_position_x': event['start_position'][0] if event['start_position'] else '',
                    'start_position_y': event['start_position'][1] if event['start_position'] else '',
                    'end_position_x': event['end_position'][0] if event['end_position'] else '',
                    'end_position_y': event['end_position'][1] if event['end_position'] else ''
                }
                
                writer.writerow(row)
                
        print(f"Exported {len(self.events)} events to {output_path}")


