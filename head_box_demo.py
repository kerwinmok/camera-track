import argparse
from typing import List, Tuple, Union

import cv2


def parse_source(value: str) -> Union[int, str]:
    # allow camera index or file path
    if value.isdigit():
        return int(value)
    return value


def clamp(val: int, low: int, high: int) -> int:
    # keep values inside a safe range
    return max(low, min(val, high))


def compute_head_box(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    # expand face bounds into a head-focused box
    hx = int(x - 0.10 * w)
    hy = int(y - 0.35 * h)
    hw = int(1.20 * w)
    hh = int(1.60 * h)

    hx = clamp(hx, 0, frame_w - 1)
    hy = clamp(hy, 0, frame_h - 1)
    hw = clamp(hw, 1, frame_w - hx)
    hh = clamp(hh, 1, frame_h - hy)
    return hx, hy, hw, hh


def interpolate_box(
    previous: Tuple[int, int, int, int],
    current: Tuple[int, int, int, int],
    alpha: float,
) -> Tuple[int, int, int, int]:
    # smooth movement between old and new box positions
    px, py, pw, ph = previous
    cx, cy, cw, ch = current
    return (
        int(px * (1.0 - alpha) + cx * alpha),
        int(py * (1.0 - alpha) + cy * alpha),
        int(pw * (1.0 - alpha) + cw * alpha),
        int(ph * (1.0 - alpha) + ch * alpha),
    )


def area(box: Tuple[int, int, int, int]) -> int:
    # prefer larger faces in single-person camera view
    return box[2] * box[3]


def detect_profile_faces(
    gray,
    profile_cascade: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
) -> List[Tuple[int, int, int, int]]:
    # detect side faces in normal orientation
    direct_profiles = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(40, 40),
    )

    # detect side faces in mirrored orientation for the opposite direction
    flipped_gray = cv2.flip(gray, 1)
    flipped_profiles = profile_cascade.detectMultiScale(
        flipped_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(40, 40),
    )

    # map mirrored detections back to original coordinates
    frame_w = gray.shape[1]
    profiles: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in direct_profiles:
        profiles.append((int(x), int(y), int(w), int(h)))
    for (x, y, w, h) in flipped_profiles:
        mapped_x = frame_w - int(x) - int(w)
        profiles.append((mapped_x, int(y), int(w), int(h)))

    return profiles


def main() -> None:
    # parse runtime options
    parser = argparse.ArgumentParser(
        description="Draw white head boxes on detected people in a video source."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index like 0/1 or a video file path. Defaults to 0.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Haar cascade scale factor. Lower is slower and can detect more.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbors for detection confidence.",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=12,
        help="Keep last head box this many frames when detection drops.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.45,
        help="Tracking smoothness from 0.0 to 1.0 (higher reacts faster).",
    )
    args = parser.parse_args()

    # open camera stream or video file
    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    # load frontal and profile detectors
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Could not load haarcascade_frontalface_default.xml")

    profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
    profile_cascade = cv2.CascadeClassifier(profile_path)
    if profile_cascade.empty():
        raise RuntimeError("Could not load haarcascade_profileface.xml")

    print("running head box demo")
    print("press q to quit")

    # tracking state for smoothing and temporary hold
    smooth_alpha = clamp(int(args.smooth_alpha * 1000), 0, 1000) / 1000.0
    tracked_box: Union[Tuple[int, int, int, int], None] = None
    frames_since_seen = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect frontal and side-facing faces
        frontal_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=(40, 40),
        )
        profile_faces = detect_profile_faces(
            gray,
            profile_cascade,
            args.scale_factor,
            args.min_neighbors,
        )

        # convert face boxes to head boxes
        frame_h, frame_w = frame.shape[:2]
        all_faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in frontal_faces]
        all_faces.extend(profile_faces)
        candidate_boxes = [
            compute_head_box(x, y, w, h, frame_w, frame_h)
            for (x, y, w, h) in all_faces
        ]

        if candidate_boxes:
            # use the largest visible face as the active track
            current_box = max(candidate_boxes, key=area)
            if tracked_box is None:
                tracked_box = current_box
            else:
                tracked_box = interpolate_box(tracked_box, current_box, smooth_alpha)
            frames_since_seen = 0
        else:
            # keep box briefly when detections flicker
            frames_since_seen += 1
            if frames_since_seen > args.hold_frames:
                tracked_box = None

        if tracked_box is not None:
            # draw white tracking box
            hx, hy, hw, hh = tracked_box
            cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (255, 255, 255), 2)

        status = "tracking" if tracked_box is not None else "waiting"
        if tracked_box is not None and frames_since_seen > 0:
            status = f"tracking (hold {frames_since_seen}/{args.hold_frames})"

        cv2.putText(
            frame,
            f"heads detected: {len(all_faces)} | {status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # show frame and allow quick exit
        cv2.imshow("camera track - head box demo", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
