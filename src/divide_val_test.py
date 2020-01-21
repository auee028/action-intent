import os
import json

val_anno_prefix = "/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/Dataset/Dense_VTT/annotations/anno_modified_1~35sec"
val_1_anno_file = os.path.join(val_anno_prefix, "val_1_short_data.json")
val_2_anno_file = os.path.join(val_anno_prefix, "val_2_short_data.json")

val_1_frames_prefix = "/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/Dataset/Dense_VTT/sampled_frames/val_1"

with open(val_1_anno_file, "r") as f1:
    contents = json.load(f1)
    print(contents.keys())