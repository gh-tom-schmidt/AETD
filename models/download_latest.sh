#!/bin/bash

# get the folder where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

urls=(
    "https://arcxyon.com/wp-content/uploads/2025/08/yolo-detect-m_best_epochs-100_size-460-960_05-08-2025.zip"
    "https://arcxyon.com/wp-content/uploads/2025/08/yolo-cls-s_best_epochs-30_size-32-32_06-08-2025.zip"
    "https://arcxyon.com/wp-content/uploads/2025/08/yolo-seg-m_full-road_best_epochs-300_size-460-960_07-08-2025.zip"
)

for url in "${urls[@]}"; do
    filename=$(basename "$url" .zip)
    zipfile="$SCRIPT_DIR/$filename.zip"

    # download into script directory
    wget -O "$zipfile" "$url"

    # unzip inside script directory
    unzip -d "$SCRIPT_DIR" "$zipfile"

    # move .pt file to models/pretrained/
    mv "$SCRIPT_DIR/$filename/$filename.pt" "$SCRIPT_DIR/pretrained/"

    # cleanup
    rm -rf "$zipfile" "$SCRIPT_DIR/$filename"
done

echo "All models downloaded to $SCRIPT_DIR/pretrained/"
