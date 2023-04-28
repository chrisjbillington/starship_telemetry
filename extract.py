from pathlib import Path
import time
import re
import concurrent.futures

from PIL import Image
import numpy as np
import pytesseract


def pmap(func, *allargs, **allkwargs):
    """Parallel (threads) map function"""
    futures = []    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(len(allargs[0])):
            args = [a[i] for a in allargs]
            kwargs = {k: v[i] for k, v in allkwargs.items()}
            future = executor.submit(func, *args, **kwargs)
            futures.append(future)
        return [future.result() for future in futures]


fields = ['timestamp', 'ship', 'booster']

TIMESTAMP_PATTERN = r'T([+-]\d{2}:\d{2}:\d{2})'
SPEED_PATTERN = r'SPEED\s+(\d+)\s*KM'
ALTITUDE_PATTERN = r'ALTITUDE\s+(\d+)\s*KM'

TIMESTAMP_TESSCONFIG = "-c tessedit_char_whitelist=0123456789T+-:"
TELEMETRY_TESSCONFIG = "-c tessedit_char_whitelist=0123456789SPEEDALTITUDEKM/H"


def cleanup(im):
    """Cleanup an image by subtracting its median pixel value, then scaling it up until
    its max on each channel is 255"""
    arr = np.array(im, dtype=float)
    if arr.mean(axis=-1).max() < 200:
        # no text here, return black image
        arr[:] = 0
    else:
        arr -= np.median(arr) + 5
        arrmax = arr.max(axis=(0, 1))
        if all(arrmax != 0):
            arr *= 255 / arrmax
        arr = arr.clip(0, 255)
    return Image.fromarray(arr.astype(np.uint8)).convert('L')


OUTFILE = Path('telemetry.csv')
OUTFILE.write_text("t,v1,h1,v2,h2\n", "utf8")

# i = 1
i = 2700 # just before launch
while True:
    # This script is designed to just keep running until the next frame is available.
    # Idea was so that it could be part of a data processing pipeline during a live
    # launch, but I probably won't do that. Anyway that's why this loop never ends.
    while not all([Path(f'{field}_frames/{i:06d}.png').exists() for field in fields]):
        time.sleep(0.1)

    # Perform OCR using Tesseract
    timestamp_text, booster_text, ship_text = pmap(
        pytesseract.image_to_string,
        [
            cleanup(Image.open(f'timestamp_frames/{i:06d}.png')),
            cleanup(Image.open(f'booster_frames/{i:06d}.png')),
            cleanup(Image.open(f'ship_frames/{i:06d}.png')),
        ],
        config=[TIMESTAMP_TESSCONFIG, TELEMETRY_TESSCONFIG, TELEMETRY_TESSCONFIG],
    )

    match = re.search(TIMESTAMP_PATTERN, timestamp_text)
    if match is not None:
        timestamp = match.groups()[0]
    else:
        timestamp = None

    match = re.search(SPEED_PATTERN, booster_text)
    if match is not None:
        booster_speed = match.groups()[0]
    else:
        booster_speed = np.nan

    match = re.search(ALTITUDE_PATTERN, booster_text)
    if match is not None:
        booster_altitude = match.groups()[0]
    else:
        booster_altitude = np.nan

    match = re.search(SPEED_PATTERN, ship_text)
    if match is not None:
        ship_speed = match.groups()[0]
    else:
        ship_speed = np.nan

    match = re.search(ALTITUDE_PATTERN, ship_text)
    if match is not None:
        ship_altitude = match.groups()[0]
    else:
        ship_altitude = np.nan

    if timestamp is not None:
        sign = +1 if timestamp[0] == '+' else -1
        h, m, s = [int(n) for n in timestamp[1:].split(':')]
        t = sign * (3600 * h + 60 * m + s)
    else:
        t = np.nan

    print(i, timestamp, t, booster_speed, booster_altitude, ship_speed, ship_altitude)

    with OUTFILE.open('a', encoding="utf8") as f:
        f.write(
            f'{t},{booster_speed},{booster_altitude},{ship_speed},{ship_altitude}\n'
        )

    i += 1
