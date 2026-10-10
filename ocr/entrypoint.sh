#!/bin/sh
# Temporary Python fallback; the selected service owns every OCR route.
set -eu
case "${NOEVIA_OCR_IMPL-python}" in
  python) exec python /app/server.py ;;
  rust)
    if ! features=$(/usr/local/bin/noevia-ocr --features); then
      echo "OCR startup refused: native-ocr feature probe failed" >&2
      exit 1
    fi
    if ! printf '%s\n' "$features" | grep -qx native-ocr; then
      echo "OCR startup refused: native-ocr support required" >&2
      exit 1
    fi
    exec /usr/local/bin/noevia-ocr
    ;;
  *) echo "OCR startup refused: NOEVIA_OCR_IMPL must be python or rust" >&2; exit 1 ;;
esac
