import argparse
import datetime
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="View and record the MaixCAM MJPEG stream."
    )
    parser.add_argument(
        "--ip",
        required=True,
        help="MaixCAM IP address, for example 192.168.1.23",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="Directory used for saved MP4 recordings.",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Display the stream without recording it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    stream_url = f"http://{args.ip}:{args.port}/stream"
    output_dir = Path(args.output_dir)
    writer = None

    capture = cv2.VideoCapture(stream_url)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open stream: {stream_url}")

    print(f"Connected: {stream_url}")
    print("Press Q to stop.")

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Stream disconnected.")
            break

        if writer is None and not args.view_only:
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"steel_ball_{stamp}.mp4"
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create recording: {output_path}")
            print(f"Recording: {output_path.resolve()}")

        if writer is not None:
            writer.write(frame)

        cv2.imshow("MaixCAM Steel Ball", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
