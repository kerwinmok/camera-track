# camera-track

simple local demo that reads a video source and draws a white box around detected heads.

the app now keeps tracking a head box smoothly while your head is visible and holds the box briefly when detection drops for a few frames.

this is detection only.
it does not move the mouse, click, control input, or automate anything.

## requirements

- python 3.9+
- pip

## install

```bash
pip install -r requirements.txt
```

## run with laptop camera

```bash
python head_box_demo.py --source 0
```

for smoother live webcam tracking:

```bash
python head_box_demo.py --source 0 --hold-frames 15 --smooth-alpha 0.5
```

## run with a video file

```bash
python head_box_demo.py --source path/to/video.mp4
```

## controls

- press `q` to exit

## tuning

- `--scale-factor` smaller values can detect more but run slower
- `--min-neighbors` larger values reduce false positives but can miss detections
- `--hold-frames` keeps the last box visible briefly when your head is missed
- `--smooth-alpha` controls box motion smoothing (higher = faster movement)

example:

```bash
python head_box_demo.py --source 0 --scale-factor 1.05 --min-neighbors 6
```
