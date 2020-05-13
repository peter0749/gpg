#!/bin/bash
# Bash version>=4.0

PATH_TO_PCDS="/xxx/GraspDataset/YCB/all"
OUTPUT_DIR_MERGED="/xxx/clouds"
OUTPUT_DIR_PROPOSAL="/xxx/proposal"
GPG_ROOT="/xxx/gpg"
NULL_POSE_PATH="$GPG_ROOT/null_registration.pose"
SAMPLER_PATH="$GPG_ROOT/build/generate_samples"
SAMPLER_CONFIG_FILE="$GPG_ROOT/cfg/params.cfg"
TARGET_NUM_VIEWS=5

mkdir -p "$OUTPUT_DIR_MERGED"
mkdir -p "$OUTPUT_DIR_PROPOSAL"

ls $PATH_TO_PCDS | while read obj; do
    # Find views for this object and shuffle them.
    mkdir "$OUTPUT_DIR_PROPOSAL/$obj"
    mkdir -p "$OUTPUT_DIR_MERGED/$obj/clouds"
    TEMP_FILE_LIST=`mktemp`
    find "$PATH_TO_PCDS/$obj/clouds/" -type f -iname '*.pcd' | sort -R > "$TEMP_FILE_LIST"
    readarray -t array < "$TEMP_FILE_LIST"
    num_views_total="${#array[@]}"
    num_processed=0
    for ((i=0; i<$num_views_total-$TARGET_NUM_VIEWS-1; i+=$TARGET_NUM_VIEWS)); do
        subarray=${array[@]:$i:$TARGET_NUM_VIEWS}
        "$SAMPLER_PATH" "$SAMPLER_CONFIG_FILE" ${subarray[@]} "$NULL_POSE_PATH" "$OUTPUT_DIR_PROPOSAL/$obj/merged-$num_processed.pose" "$OUTPUT_DIR_MERGED/$obj/clouds/merged-$num_processed.pcd"
        num_processed=$((num_processed+1))
    done
done
