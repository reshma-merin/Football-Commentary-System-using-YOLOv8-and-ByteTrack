# Football Commentary System

## OVERVIEW
This project implements an automated system that analyzes football match videos to:
- Detect players, referees, and the ball using a **fine-tuned YOLO model trained on a Roboflow football dataset**
- Track objects over time using ByteTrack tracking algorithm
- Assign teams to players by clustering jersey colors using KMeans
- Estimate camera movements to adjust player tracking positions
- Transform player positions from pixel coordinates to real-world field coordinates
- Detect key football events such as passes, goals, contested possessions, and loss of possession
- Generate human-like commentary text for detected events
- Convert the commentary text into audio using Google Text-to-Speech (gTTS)
- Save an annotated output video with bounding boxes, event labels, and commentary overlays


---

## Project Structure

| File/Folder | Description |
|-------------|-------------|
| `main.py` | Main pipeline integrating all components to produce final video and audio commentary |
| `yolo_inference.py` | Runs YOLO model inference on input videos |
| `commentary_generator.py` | Creates text-based football commentary based on detected events |
| `text_to_speech.py` | Converts generated commentary text into an MP3 audio file |
| `football_event_detector.py` | Detects advanced football events like passes, goals, duels based on tracking data |
| `event_detector.py` | Detects simple possession-based events (pass, lost possession, contested ball) |
| `tracker.py` | Object tracking of players, referees, and ball using ByteTrack and YOLO |
| `view_transformer.py` | Converts image coordinates to real-world field coordinates |
| `team_assigner.py` | Assigns players to teams by analyzing jersey colors using KMeans clustering |
| `speed_and_distance_estimator.py` | Estimates player speeds and distances covered |
| `player_ball_assigner.py` | Assigns ball possession to nearest player |
| `camera_movement_estimator.py` | Estimates and adjusts for camera motion between frames |
| `video_utils.py` | Utilities to read and save videos |
| `bbox_utils.py` | Bounding box utilities: center, width, foot position calculations |
| `__init__.py` | Initializes utility imports |

---

## Installation

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Download or train a YOLO model (`best.pt`) for detecting players, referees, and balls.

3. Place your input video under the `input_videos/` folder.

---

## Usage

To run the complete pipeline:

```bash
python main.py
```


## Main Steps Executed

- Video frames extraction
- Object detection & tracking
- Camera motion compensation
- Coordinate transformation
- Team assignment
- Speed & distance calculation
- Event detection
- Commentary generation (text)
- Commentary generation (audio)
- Annotated video creation

---

## Output

- Annotated video: `output_videos/output_video_1.avi`
- Commentary audio: `audio_commentaries/full_match_commentary.mp3`

---

## Notes

- Ensure the YOLO model (`best.pt`) is compatible with football players, referees, and ball detection.
- Stubs (`track_stubs.pkl`, `camera_movement_stub.pkl`) can be used to save time during repeated runs.
- Audio commentary is generated using Google TTS (gTTS) in English.

---

## Future Improvements

- More detailed event classification (e.g., fouls, offsides)
- Real-time live commentary during live match feeds
- Support for multi-language commentary
- Improve voice naturalness using advanced TTS models like Tacotron or Bark

---

## Credits

- YOLO Object Detection (Ultralytics)
- ByteTrack for object tracking
- gTTS for Text-to-Speech
- Inspired by academic approaches to football analytics





<sub>KHUSHI MAHAJAN ,           RESHMA MERIN THOMAS ,          SHEILEY PATEL,           ANVITHA REDDY THUPALLLY</sub>

