#!/bin/zsh
set -e

ARIA_ENV="/Users/dinithidissanayake/Documents/Aria 2/.venv-gen2"
export PATH="$ARIA_ENV/bin:$PATH"

# Verify the glasses are connected by USB.
DEVICE_OUTPUT=$(aria_gen2 device list 2>&1)
print -r -- "$DEVICE_OUTPUT"
if [[ "$DEVICE_OUTPUT" == *"No devices found"* ]]; then
  print "\nThe Mac can see the USB cable, but the Aria SDK cannot discover the glasses."
  print "Wake the glasses and enable USB networking, then run this command again."
  read "?Press Enter to close."
  exit 1
fi

# Streaming must be active before the viewer opens.
aria_gen2 streaming start

# Stop streaming automatically when the viewer closes or this command exits.
stop_stream() {
  aria_gen2 streaming stop || true
}
trap stop_stream EXIT INT TERM

aria_streaming_viewer --real-time --interpolate --rerun-memory-limit 4GB
