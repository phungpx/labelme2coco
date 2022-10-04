#!/usr/bin/env python

import argparse
import collections
import datetime
import json
import os
import os.path as osp
from pathlib import Path
import sys
import uuid

import imgviz
import numpy as np

import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


group_categories = {
    'CARD': [
        'BLX',
        'BLX_BACK',
        'BLX_OLD',
        'BLX_BACK_OLD',
        'CMND',
        'CMND_BACK',
        'CCCD',
        'CCCD_BACK',
        'CMCC',
        'CCCD_front_chip',
        'CCCD_back_chip',
        'CMQD_A',
        'CMQD_A_BACK',
        'CMQD_B',
        'CMQD_B_BACK',
        'CMQD_C',
        'CMQD_C_BACK',
        'CMQD_D',
        'CMQD_D_BACK',
        'CMQD_B_VT',
        'CMQD_B_VT_BACK',
        'PASSPORT',
        'PASSPORT_OTHER',
    ]
}

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", help="input annotated directory")
    parser.add_argument("--pattern", default='**/*.json')
    parser.add_argument("--output-dir", help="output dataset directory")
    parser.add_argument("--set-name", default='train')
    parser.add_argument("--labels", help="labels file", required=True, default='labels.txt')
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()

    output_dir = f'{args.output_dir}/{args.set_name}'
    if osp.exists(output_dir):
        print("Output directory already exists:", output_dir)
        # sys.exit(1)
    os.makedirs(output_dir)
    os.makedirs(osp.join(output_dir, "JPEGImages"))
    os.makedirs(osp.join(output_dir, "JPEGImagesErrors"))

    if not args.noviz:
        os.makedirs(osp.join(output_dir, "Visualization"))
    print("Creating dataset:", output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,)
        )

    out_ann_file = osp.join(output_dir, "annotations.json")

    # label_files = list(Path(args.input_dir).glob(args.pattern))
    # ----------get all dataset------------------------------------
    label_files = []
    for card_type in group_categories['CARD']:
        label_files.extend(list(Path(args.input_dir).joinpath(card_type, args.set_name).glob(args.pattern)))
    print(f'The number of labels: {len(label_files)}')
    # -------------------------------------------------------------

    for image_id, filename in enumerate(label_files):
        try:
            print("Generating dataset from:", filename)

            filename = str(filename)
            label_file = labelme.LabelFile(filename=filename)

            base = osp.splitext(osp.basename(filename))[0]
            out_img_file = osp.join(output_dir, "JPEGImages", base + ".jpg")

            img = labelme.utils.img_data_to_arr(label_file.imageData)
            imgviz.io.imsave(out_img_file, img)
            data["images"].append(
                dict(
                    license=0,
                    url=None,
                    file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                )
            )

            masks = {}  # for area
            segmentations = collections.defaultdict(list)  # for segmentation
            for shape in label_file.shapes:
                points = shape["points"]
                label = shape["label"]
                group_id = shape.get("group_id")
                shape_type = shape.get("shape_type", "polygon")
                mask = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)

                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask

                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                else:
                    points = np.asarray(points).flatten().tolist()

                segmentations[instance].append(points)
            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance

                # custom ------------------------
                cls_id = None
                for main_category, category_id in class_name_to_id.items():
                    if cls_name in group_categories.get(main_category, []):
                        cls_id = category_id
                        break

                if cls_id is None:
                    continue
                # ------------------------------

                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data["annotations"].append(
                    dict(
                        id=len(data["annotations"]),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )

            if not args.noviz:
                # --- custom ----------------------------------------------
                _labels, _captions, _masks = [], [], []
                for (cnm, _), msk in masks.items():
                    for group_name, class_names in group_categories.items():
                        if cnm in class_names:
                            _labels.append(class_name_to_id[group_name])
                            _captions.append(cnm)
                            _masks.append(msk)
                            break
                # ---------------------------------------------------------
                # # --- original --------------------------------------------
                # labels, captions, masks = zip(
                #     *[
                #         (class_name_to_id[cnm], cnm, msk)
                #         for (cnm, gid), msk in masks.items()
                #         if cnm in class_name_to_id
                #     ]
                # )
                # # ---------------------------------------------------------
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=_labels,
                    masks=_masks,
                    captions=_captions,
                    font_size=15,
                    line_width=2,
                )
                out_viz_file = osp.join(output_dir, "Visualization", base + ".jpg")
                imgviz.io.imsave(out_viz_file, viz)
        except:
            print(filename)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
