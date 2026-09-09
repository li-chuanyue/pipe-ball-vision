import os

from maix import app, camera, display, image, key, time


SAVE_DIR = "/root/rocker_pipe_ball_capture/images"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 448
JPEG_QUALITY = 95
FILE_PREFIX = "rocker_pipe_ball_"

WHITE = image.Color.from_rgb(255, 255, 255)
GREEN = image.Color.from_rgb(0, 255, 0)
BLACK = image.Color.from_rgb(0, 0, 0)

capture_requested = False
saved_count = 0
next_index = 1
notice_until_ms = 0
last_saved_name = ""


def scan_saved_images():
    count = 0
    maximum = 0

    for filename in os.listdir(SAVE_DIR):
        if not filename.startswith(FILE_PREFIX) or not filename.endswith(".jpg"):
            continue

        number_text = filename[len(FILE_PREFIX) : -4]
        try:
            number = int(number_text)
        except ValueError:
            continue

        count += 1
        maximum = max(maximum, number)

    return count, maximum + 1


def on_key(key_id, state):
    global capture_requested

    if key_id != key.Keys.KEY_OK:
        return

    if state == key.State.KEY_LONG_PRESSED:
        app.set_exit_flag(True)
    elif state == key.State.KEY_RELEASED:
        capture_requested = True


os.makedirs(SAVE_DIR, exist_ok=True)
saved_count, next_index = scan_saved_images()

cam = camera.Camera(IMAGE_WIDTH, IMAGE_HEIGHT)
disp = display.Display()
key_input = key.Key(on_key, long_press_time=2000)
cam.skip_frames(30)

print("Dataset capture started")
print("Save directory:", SAVE_DIR)
print("Resolution:", IMAGE_WIDTH, "x", IMAGE_HEIGHT)
print("Short press USER/OK: capture")
print("Long press USER/OK: exit")
print("Existing images:", saved_count)

while not app.need_exit():
    img = cam.read()

    if capture_requested:
        capture_requested = False
        filename = "{}{:06d}.jpg".format(FILE_PREFIX, next_index)
        save_path = os.path.join(SAVE_DIR, filename)
        img.save(save_path, quality=JPEG_QUALITY)

        saved_count += 1
        next_index += 1
        last_saved_name = filename
        notice_until_ms = time.ticks_ms() + 800
        print("Saved:", save_path)

    img.draw_rect(0, 0, IMAGE_WIDTH, 54, BLACK, thickness=-1)
    img.draw_string(8, 5, "Short press: SAVE  Long press: EXIT", WHITE)
    img.draw_string(8, 29, "Saved: {}".format(saved_count), GREEN)

    if time.ticks_ms() < notice_until_ms:
        img.draw_string(8, 60, "SAVED: {}".format(last_saved_name), GREEN)

    disp.show(img)
