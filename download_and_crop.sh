#!/bin/bash

VIDEO_ID=-1wcilQ58hI

# Booster region
X_POS_BOOSTER=130
WIDTH_BOOSTER=230
Y_POS_BOOSTER=592
HEIGHT_BOOSTER=64

# Ship region
X_POS_SHIP=920
WIDTH_SHIP=230
Y_POS_SHIP=592
HEIGHT_SHIP=64

# Timestamp region
X_POS_TIMESTAMP=560
WIDTH_TIMESTAMP=160
Y_POS_TIMESTAMP=618
HEIGHT_TIMESTAMP=48

# Output frame rate
FRAMRATE=1
rm -rf booster_frames ship_frames timestamp_frames frames
mkdir -p booster_frames ship_frames timestamp_frames frames

yt-dlp -f "best[height<=720]" --hls-use-mpegts -o - "https://www.youtube.com/watch?v=${VIDEO_ID}" | \
ffmpeg -i pipe:0 \
       -r ${FRAMRATE} frames/%06d.png \
       -vf "crop=${WIDTH_BOOSTER}:${HEIGHT_BOOSTER}:${X_POS_BOOSTER}:${Y_POS_BOOSTER}" -r ${FRAMRATE} booster_frames/%06d.png \
       -vf "crop=${WIDTH_SHIP}:${HEIGHT_SHIP}:${X_POS_SHIP}:${Y_POS_SHIP}" -r ${FRAMRATE} ship_frames/%06d.png \
       -vf "crop=${WIDTH_TIMESTAMP}:${HEIGHT_TIMESTAMP}:${X_POS_TIMESTAMP}:${Y_POS_TIMESTAMP}" -r ${FRAMRATE} timestamp_frames/%06d.png
