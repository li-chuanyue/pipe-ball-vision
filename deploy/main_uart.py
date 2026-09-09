import struct

from maix import app, camera, display, image, nn, pinmap, uart


MODEL_PATH = "/root/models/pipe_ball_final_v6/pipe_ball_final_v6.mud"

BALL_CLASS_ID = 0
PIPE_CLASS_ID = 1
DETECT_THRESHOLD = 0.20
BALL_CONF_THRESHOLD = 0.22
PIPE_CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

BALL_COLOR = image.Color.from_rgb(255, 40, 40)
PIPE_COLOR = image.Color.from_rgb(40, 255, 80)
CENTER_COLOR = image.Color.from_rgb(255, 220, 0)
TEXT_COLOR = image.Color.from_rgb(255, 255, 255)

UART_DEVICE = "/dev/ttyS1"
UART_BAUDRATE = 115200
UART_TX_PIN = "A19"
UART_RX_PIN = "A18"

PACKET_HEADER_1 = 0xAA
PACKET_HEADER_2 = 0x55
PACKET_VERSION = 0x02

FLAG_BALL_VALID = 0x01
FLAG_PIPE_VALID = 0x02
FLAG_BALL_FRESH = 0x04
FLAG_PIPE_FRESH = 0x08


class SmoothBox:
    def __init__(self, alpha, hold_frames):
        self.alpha = alpha
        self.hold_frames = hold_frames
        self.valid = False
        self.detected = False
        self.missed = 0
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0
        self.score = 0.0

    def update(self, obj):
        self.detected = obj is not None

        if obj is None:
            self.missed += 1
            if self.missed > self.hold_frames:
                self.valid = False
            return

        if not self.valid:
            self.x = float(obj.x)
            self.y = float(obj.y)
            self.w = float(obj.w)
            self.h = float(obj.h)
        else:
            keep = 1.0 - self.alpha
            self.x = keep * self.x + self.alpha * obj.x
            self.y = keep * self.y + self.alpha * obj.y
            self.w = keep * self.w + self.alpha * obj.w
            self.h = keep * self.h + self.alpha * obj.h

        self.score = obj.score
        self.missed = 0
        self.valid = True

    def box(self):
        return (
            int(self.x + 0.5),
            int(self.y + 0.5),
            int(self.w + 0.5),
            int(self.h + 0.5),
        )

    def center(self):
        x, y, w, h = self.box()
        return x + w // 2, y + h // 2


def best_objects(objects):
    ball = None
    pipe = None

    for obj in objects:
        if (
            obj.class_id == BALL_CLASS_ID
            and obj.score >= BALL_CONF_THRESHOLD
        ):
            if ball is None or obj.score > ball.score:
                ball = obj
        elif (
            obj.class_id == PIPE_CLASS_ID
            and obj.score >= PIPE_CONF_THRESHOLD
        ):
            if pipe is None or obj.score > pipe.score:
                pipe = obj

    return ball, pipe


def draw_track(img, track, label, color):
    x, y, w, h = track.box()
    center_x, center_y = track.center()
    img.draw_rect(x, y, w, h, color, thickness=2)
    img.draw_cross(center_x, center_y, color, size=5, thickness=2)

    state = "" if track.detected else " HOLD"
    img.draw_string(
        x,
        max(0, y - 14),
        "{} {:.2f}{}".format(label, track.score, state),
        color,
    )
    return center_x, center_y


def crc8(data):
    crc = 0
    for value in data:
        crc ^= value
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def packet_coordinate(track):
    if not track.valid:
        return 0xFFFF, 0xFFFF
    center_x, center_y = track.center()
    center_x = max(0, min(0xFFFE, center_x))
    center_y = max(0, min(0xFFFE, center_y))
    return center_x, center_y


def make_uart_packet(sequence, ball_track, pipe_track):
    flags = 0
    if ball_track.valid:
        flags |= FLAG_BALL_VALID
    if pipe_track.valid:
        flags |= FLAG_PIPE_VALID
    if ball_track.detected:
        flags |= FLAG_BALL_FRESH
    if pipe_track.detected:
        flags |= FLAG_PIPE_FRESH

    ball_x, ball_y = packet_coordinate(ball_track)
    pipe_x, pipe_y = packet_coordinate(pipe_track)
    confidence = 0
    if ball_track.valid:
        confidence = max(0, min(100, int(ball_track.score * 100 + 0.5)))

    packet_without_crc = struct.pack(
        "<BBBBHHHHHB",
        PACKET_HEADER_1,
        PACKET_HEADER_2,
        PACKET_VERSION,
        flags,
        sequence,
        ball_x,
        ball_y,
        pipe_x,
        pipe_y,
        confidence,
    )
    return packet_without_crc + bytes([crc8(packet_without_crc[2:])])


def main():
    pinmap.set_pin_function(UART_RX_PIN, "UART1_RX")
    pinmap.set_pin_function(UART_TX_PIN, "UART1_TX")
    serial = uart.UART(UART_DEVICE, UART_BAUDRATE)

    # Synchronous inference keeps detections aligned with the current frame.
    detector = nn.YOLO11(model=MODEL_PATH, dual_buff=False)
    cam = camera.Camera(
        detector.input_width(),
        detector.input_height(),
        detector.input_format(),
        buff_num=1,
    )
    disp = display.Display()
    cam.skip_frames(30)

    print("Model:", MODEL_PATH)
    print("Mode: LOW_LATENCY_V2")
    print("Input:", detector.input_width(), "x", detector.input_height())
    print("Labels:", detector.labels)
    print(
        "UART:",
        UART_DEVICE,
        UART_BAUDRATE,
        "TX=" + UART_TX_PIN,
        "RX=" + UART_RX_PIN,
    )

    frame_count = 0
    sequence = 0
    # Pipe coordinates can be smoothed because the pipe changes slowly.
    pipe_track = SmoothBox(alpha=0.30, hold_frames=3)
    # Do not smooth or hold the ball: stale coordinates are unsafe for control.
    ball_track = SmoothBox(alpha=1.00, hold_frames=0)

    while not app.need_exit():
        # Drop any queued image and wait for the newest camera frame.
        cam.clear_buff()
        img = cam.read()
        objects = detector.detect(
            img,
            conf_th=DETECT_THRESHOLD,
            iou_th=IOU_THRESHOLD,
        )
        ball, pipe = best_objects(objects)
        ball_track.update(ball)
        pipe_track.update(pipe)

        # Send coordinates before drawing or preview transmission.
        packet = make_uart_packet(sequence, ball_track, pipe_track)
        serial.write(packet)
        sequence = (sequence + 1) & 0xFFFF

        ball_center = None
        if ball_track.valid:
            ball_center = draw_track(
                img,
                ball_track,
                "steel_ball",
                BALL_COLOR,
            )

        if pipe_track.valid:
            draw_track(
                img,
                pipe_track,
                "pipe",
                PIPE_COLOR,
            )

        if ball_track.valid and pipe_track.valid and pipe_track.w > 0:
            ball_x, ball_y = ball_center
            pipe_x, pipe_y, pipe_w, pipe_h = pipe_track.box()
            pipe_left = pipe_x

            position = (ball_x - pipe_left) / pipe_w
            position = max(0.0, min(1.0, position))
            position_percent = position * 100.0
            center_error_percent = (position - 0.5) * 200.0

            img.draw_cross(
                pipe_x + pipe_w // 2,
                pipe_y + pipe_h // 2,
                CENTER_COLOR,
                size=7,
                thickness=2,
            )
            img.draw_string(
                2,
                2,
                "POS:{:.1f}% ERR:{:+.1f}%".format(
                    position_percent,
                    center_error_percent,
                ),
                CENTER_COLOR,
            )

            if frame_count % 30 == 0:
                print(
                    "ball=({},{}), pos={:.1f}%, center_err={:+.1f}%, "
                    "ball_conf={:.3f}, pipe_conf={:.3f}".format(
                        ball_x,
                        ball_y,
                        position_percent,
                        center_error_percent,
                        ball_track.score,
                        pipe_track.score,
                    )
                )
        else:
            missing = []
            if not ball_track.valid:
                missing.append("ball")
            if not pipe_track.valid:
                missing.append("pipe")
            img.draw_string(
                2,
                2,
                "WAIT: " + ",".join(missing),
                TEXT_COLOR,
            )

        frame_count += 1
        disp.show(img)


main()
