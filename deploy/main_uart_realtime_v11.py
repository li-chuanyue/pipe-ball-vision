import struct

from maix import app, camera, display, image, nn, pinmap, time, uart


MODEL_PATH = "/root/models/pipe_ball_final_v6/pipe_ball_final_v6.mud"

BALL_CLASS_ID = 0
PIPE_CLASS_ID = 1
DETECT_THRESHOLD = 0.20
BALL_CONF_THRESHOLD = 0.22
PIPE_CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

BALL_COLOR = image.Color.from_rgb(255, 40, 40)
RULER_COLOR = image.Color.from_rgb(0, 190, 255)
ZERO_COLOR = image.Color.from_rgb(255, 220, 0)
TEXT_COLOR = image.Color.from_rgb(255, 255, 255)
MASK_COLOR = image.Color.from_rgb(0, 0, 0)

UART_DEVICE = "/dev/ttyS1"
UART_BAUDRATE = 115200
UART_TX_PIN = "A19"
UART_RX_PIN = "A18"

PACKET_HEADER_1 = 0xAA
PACKET_HEADER_2 = 0x55
PACKET_VERSION = 0x04

# Approximate camera capture plus inference latency.
PREDICT_MS = 60
MAX_PREDICT_PIXELS = 32
VELOCITY_ALPHA = 0.70

PIPE_LENGTH_MM = 250.0
CALIBRATION_FRAMES = 30
MIN_PIPE_LENGTH_PX = 60.0
CALIBRATION_OUTLIER_RATIO = 0.18
RULER_DISPLAY_Y = 112
ROI_MARGIN_X = 20
ROI_MARGIN_Y = 55
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 224
RULER_DELTA_MM_INVALID = -32768

# UART is updated after every inference. The display is intentionally slower
# so rendering cannot hold up the control loop.
DISPLAY_DIVIDER = 2


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


class BallMotionPredictor:
    def __init__(self):
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_ms = 0
        self.vx = 0.0
        self.vy = 0.0
        self.ready = False

    def update(self, ball_track, pipe_scale, now_ms):
        if not ball_track.valid or not ball_track.detected:
            self.ready = False
            return None, False

        center_x, center_y = ball_track.center()
        predicted = False

        if self.ready:
            dt_ms = now_ms - self.last_ms
            if 0 < dt_ms <= 250:
                instant_vx = (center_x - self.last_x) / dt_ms
                instant_vy = (center_y - self.last_y) / dt_ms
                keep = 1.0 - VELOCITY_ALPHA
                self.vx = keep * self.vx + VELOCITY_ALPHA * instant_vx
                self.vy = keep * self.vy + VELOCITY_ALPHA * instant_vy

                dx = self.vx * PREDICT_MS
                dy = self.vy * PREDICT_MS
                dx = max(-MAX_PREDICT_PIXELS, min(MAX_PREDICT_PIXELS, dx))
                dy = max(-MAX_PREDICT_PIXELS, min(MAX_PREDICT_PIXELS, dy))
                center_x += int(dx)
                center_y += int(dy)
                predicted = True
            else:
                self.vx = 0.0
                self.vy = 0.0

        self.last_x, self.last_y = ball_track.center()
        self.last_ms = now_ms
        self.ready = True

        center_x = max(0, min(319, center_x))
        center_y = max(0, min(223, center_y))
        if pipe_scale.ready:
            center_x = max(
                int(pipe_scale.left_x + 0.5),
                min(int(pipe_scale.right_x + 0.5), center_x),
            )

        return (center_x, center_y), predicted


class InitialPipeScale:
    def __init__(self):
        self.samples = 0
        self.ready = False
        self.left_sum = 0.0
        self.right_sum = 0.0
        self.center_y_sum = 0.0
        self.bottom_sum = 0.0
        self.length_sum = 0.0
        self.left_x = 0.0
        self.right_x = 0.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.length_px = 0.0
        self.ruler_y = 0
        self.roi_x = 0
        self.roi_y = 0
        self.roi_w = IMAGE_WIDTH
        self.roi_h = IMAGE_HEIGHT

    def update(self, pipe_track):
        if self.ready or not pipe_track.valid or not pipe_track.detected:
            return

        pipe_x, pipe_y, pipe_w, pipe_h = pipe_track.box()
        left_x = float(max(0, pipe_x + 2))
        right_x = float(min(319, pipe_x + pipe_w - 2))
        length_px = right_x - left_x
        if length_px < MIN_PIPE_LENGTH_PX:
            return

        if self.samples >= 5:
            average_length = self.length_sum / self.samples
            if (
                abs(length_px - average_length)
                > average_length * CALIBRATION_OUTLIER_RATIO
            ):
                return

        center_y = pipe_y + pipe_h * 0.5
        bottom_y = pipe_y + pipe_h
        self.left_sum += left_x
        self.right_sum += right_x
        self.center_y_sum += center_y
        self.bottom_sum += bottom_y
        self.length_sum += length_px
        self.samples += 1

        if self.samples < CALIBRATION_FRAMES:
            return

        self.left_x = self.left_sum / self.samples
        self.right_x = self.right_sum / self.samples
        self.center_x = (self.left_x + self.right_x) * 0.5
        self.center_y = self.center_y_sum / self.samples
        self.length_px = self.right_x - self.left_x
        average_bottom = self.bottom_sum / self.samples
        average_top = self.center_y * 2.0 - average_bottom
        self.ruler_y = RULER_DISPLAY_Y
        roi_right = min(
            IMAGE_WIDTH,
            int(self.right_x + ROI_MARGIN_X + 0.5),
        )
        roi_bottom = min(
            IMAGE_HEIGHT,
            int(average_bottom + ROI_MARGIN_Y + 0.5),
        )
        self.roi_x = max(0, int(self.left_x - ROI_MARGIN_X + 0.5))
        self.roi_y = max(0, int(average_top - ROI_MARGIN_Y + 0.5))
        self.roi_w = max(1, roi_right - self.roi_x)
        self.roi_h = max(1, roi_bottom - self.roi_y)
        self.ready = True
        print(
            "SCALE CALIBRATED: left={}, right={}, center=({},{}), "
            "length={:.1f}px, ruler_y={}, roi=({},{},{},{})".format(
                int(self.left_x + 0.5),
                int(self.right_x + 0.5),
                int(self.center_x + 0.5),
                int(self.center_y + 0.5),
                self.length_px,
                self.ruler_y,
                self.roi_x,
                self.roi_y,
                self.roi_w,
                self.roi_h,
            )
        )

    def contains(self, obj):
        if not self.ready:
            return True
        center_x = obj.x + obj.w * 0.5
        center_y = obj.y + obj.h * 0.5
        return (
            self.roi_x <= center_x < self.roi_x + self.roi_w
            and self.roi_y <= center_y < self.roi_y + self.roi_h
        )

    def reference_center(self):
        if not self.ready:
            return None
        return (
            int(self.center_x + 0.5),
            int(self.center_y + 0.5),
        )

    def position_mm(self, ball_center):
        if not self.ready or ball_center is None:
            return None
        ball_x, _ = ball_center
        return (
            (ball_x - self.center_x)
            * PIPE_LENGTH_MM
            / self.length_px
        )

    def ball_on_ruler(self, ball_center):
        position_mm = self.position_mm(ball_center)
        if position_mm is None:
            return None

        position_mm = max(
            -PIPE_LENGTH_MM * 0.5,
            min(PIPE_LENGTH_MM * 0.5, position_mm),
        )
        mapped_x = (
            self.center_x
            + position_mm * self.length_px / PIPE_LENGTH_MM
        )
        return int(mapped_x + 0.5), self.ruler_y

    def draw_ruler(self, img):
        if not self.ready:
            return

        left_x = int(self.left_x + 0.5)
        right_x = int(self.right_x + 0.5)
        center_x = int(self.center_x + 0.5)
        ruler_y = self.ruler_y

        img.draw_line(
            left_x,
            ruler_y,
            right_x,
            ruler_y,
            RULER_COLOR,
            thickness=2,
        )

        # One small tick per centimetre and one long tick per 5 cm.
        for mm in range(-120, 121, 10):
            tick_x = int(
                self.center_x
                + mm * self.length_px / PIPE_LENGTH_MM
                + 0.5
            )
            major = (mm % 50) == 0
            tick_size = 8 if major else 4
            tick_color = ZERO_COLOR if mm == 0 else RULER_COLOR
            img.draw_line(
                tick_x,
                ruler_y - tick_size,
                tick_x,
                ruler_y + tick_size,
                tick_color,
                thickness=2 if major else 1,
            )

            if major:
                value_cm = mm // 10
                if value_cm > 0:
                    label = "+{}".format(value_cm)
                else:
                    label = str(value_cm)
                label_x = max(0, min(300, tick_x - 7))
                label_y = max(0, ruler_y - 22)
                img.draw_string(
                    label_x,
                    label_y,
                    label,
                    tick_color,
                )

        # The full pipe is 25 cm, so the two ends are -12.5 and +12.5.
        for tick_x, label, label_offset in (
            (left_x, "-12.5", 2),
            (right_x, "+12.5", -42),
        ):
            img.draw_line(
                tick_x,
                ruler_y - 9,
                tick_x,
                ruler_y + 9,
                RULER_COLOR,
                thickness=2,
            )
            label_x = max(0, min(288, tick_x + label_offset))
            label_y = min(210, ruler_y + 10)
            img.draw_string(
                label_x,
                label_y,
                label,
                RULER_COLOR,
            )

        # Emphasize the zero point without adding another tracking marker.
        img.draw_line(
            center_x,
            ruler_y - 10,
            center_x,
            ruler_y + 10,
            ZERO_COLOR,
            thickness=2,
        )


def best_objects(objects, pipe_scale):
    ball = None
    pipe = None

    for obj in objects:
        if (
            obj.class_id == BALL_CLASS_ID
            and obj.score >= BALL_CONF_THRESHOLD
            and pipe_scale.contains(obj)
        ):
            if ball is None or obj.score > ball.score:
                ball = obj
        elif not pipe_scale.ready and (
            obj.class_id == PIPE_CLASS_ID
            and obj.score >= PIPE_CONF_THRESHOLD
        ):
            if pipe is None or obj.score > pipe.score:
                pipe = obj

    return ball, pipe


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


def packet_coordinate(track, center_override=None):
    if not track.valid:
        return 0xFFFF, 0xFFFF
    if center_override is None:
        center_x, center_y = track.center()
    else:
        center_x, center_y = center_override
    center_x = max(0, min(0xFFFE, center_x))
    center_y = max(0, min(0xFFFE, center_y))
    return center_x, center_y


def packet_ruler_delta_mm(pipe_scale, ball_center):
    if not pipe_scale.ready or ball_center is None:
        return RULER_DELTA_MM_INVALID

    position_mm = pipe_scale.position_mm(ball_center)
    if position_mm is None:
        return RULER_DELTA_MM_INVALID

    delta = int(position_mm + (0.5 if position_mm >= 0 else -0.5))
    return max(-32767, min(32767, delta))


def make_uart_packet(
    ball_center,
    pipe_scale,
):
    ruler_delta_mm = packet_ruler_delta_mm(pipe_scale, ball_center)

    packet_without_crc = struct.pack(
        "<BBBh",
        PACKET_HEADER_1,
        PACKET_HEADER_2,
        PACKET_VERSION,
        ruler_delta_mm,
    )
    return packet_without_crc + bytes([crc8(packet_without_crc[2:])])


def main():
    pinmap.set_pin_function(UART_RX_PIN, "UART1_RX")
    pinmap.set_pin_function(UART_TX_PIN, "UART1_TX")
    serial = uart.UART(UART_DEVICE, UART_BAUDRATE)

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
    print("Mode: UART_PRIORITY_RULER_V11")
    print("Pipe length:", PIPE_LENGTH_MM, "mm")
    print("Prediction:", PREDICT_MS, "ms")
    print("Input:", detector.input_width(), "x", detector.input_height())
    print("Labels:", detector.labels)
    print(
        "UART:",
        UART_DEVICE,
        UART_BAUDRATE,
        "TX=" + UART_TX_PIN,
        "RX=" + UART_RX_PIN,
    )
    print("Hold the physical pipe level during startup calibration.")

    frame_count = 0
    sequence = 0
    pipe_track = SmoothBox(alpha=0.75, hold_frames=1)
    ball_track = SmoothBox(alpha=1.00, hold_frames=0)
    ball_predictor = BallMotionPredictor()
    pipe_scale = InitialPipeScale()

    while not app.need_exit():
        img = cam.read()
        objects = detector.detect(
            img,
            conf_th=DETECT_THRESHOLD,
            iou_th=IOU_THRESHOLD,
        )

        ball, pipe = best_objects(objects, pipe_scale)
        ball_track.update(ball)
        if not pipe_scale.ready:
            pipe_track.update(pipe)
            pipe_scale.update(pipe_track)

        ball_center, ball_predicted = ball_predictor.update(
            ball_track,
            pipe_scale,
            time.ticks_ms(),
        )

        packet = make_uart_packet(
            ball_center,
            pipe_scale,
        )
        serial.write(packet)
        sequence = (sequence + 1) & 0xFFFF

        if frame_count % DISPLAY_DIVIDER == 0:
            # Rendering is lower priority than UART and happens after sending.
            img.draw_rect(
                0,
                0,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                MASK_COLOR,
                thickness=-1,
            )

            if pipe_scale.ready:
                pipe_scale.draw_ruler(img)
                ruler_ball = pipe_scale.ball_on_ruler(ball_center)
                if ball_track.valid and ruler_ball is not None:
                    img.draw_cross(
                        ruler_ball[0],
                        ruler_ball[1],
                        BALL_COLOR,
                        size=7,
                        thickness=2,
                    )
            else:
                img.draw_string(
                    68,
                    102,
                    "HOLD LEVEL CAL:{}/{}".format(
                        pipe_scale.samples,
                        CALIBRATION_FRAMES,
                    ),
                    TEXT_COLOR,
                )
            disp.show(img)

        frame_count += 1


main()
