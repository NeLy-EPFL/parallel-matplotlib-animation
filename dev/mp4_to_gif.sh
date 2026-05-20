#!/usr/bin/env bash

# Convert an MP4 video to a GIF with specified scale and optional frame limit.
# Useful for creating GIFs from rendered videos for documentation.
# Usage:
# ./mp4_to_gif.sh input.mp4 output.gif scale max_frames
# Example:
# ./mp4_to_gif.sh input.mp4 output.gif 320 100   # limit to 100 frames
# ./mp4_to_gif.sh input.mp4 output.gif 320 0     # unlimited

input_path="$1"
output_path="$2"
scale="$3"
max_frames="$4"

# Check arguments
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  echo "Usage: $0 <input_path> <output_path> <scale> [max_frames]"
  exit 1
fi

# Set frame limit option
frame_limit=""
if [ -n "$max_frames" ] && [ "$max_frames" -gt 0 ]; then
    frame_limit="-vframes $max_frames"
fi

ffmpeg -i "$input_path" $frame_limit \
-filter_complex "fps=10,scale=${scale}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=dither=sierra2_4a" \
-loop 0 "$output_path"

