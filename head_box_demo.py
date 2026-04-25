import argparse
from typing import Tuple, Union

import cv2


def parse_source(value: str) -> Union[int, str]:
    if value.isdigit():
        return int(value)
    return value


def clamp(val: int, low: int, high: int) -> int:
    return max(low, min(val, high))


def compute_head_box(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    # Expand a face rectangle into a head-focused rectangle.
    hx = int(x - 0.10 * w)
    hy = int(y - 0.35 * h)
    hw = int(1.20 * w)
    hh = int(1.60 * h)

    hx = clamp(hx, 0, frame_w - 1)
    hy = clamp(hy, 0, frame_h - 1)
    hw = clamp(hw, 1, frame_w - hx)
    hh = clamp(hh, 1, frame_h - hy)
    return hx, hy, hw, hh


def main() -> None:
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
    args = parser.parse_args()

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Could not load haarcascade_frontalface_default.xml")

    print("running head box demo")
    print("press q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=(40, 40),
        )

        frame_h, frame_w = frame.shape[:2]

        for (x, y, w, h) in faces:
            hx, hy, hw, hh = compute_head_box(x, y, w, h, frame_w, frame_h)
            cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (255, 255, 255), 2)

        cv2.putText(
            frame,
            f"heads detected: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("camera track - head box demo", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
