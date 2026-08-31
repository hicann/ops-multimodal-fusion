# Copyright (c) OpenMMLab. All rights reserved.
import sys
from pathlib import Path

import mmengine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kitti_data_utils  # noqa: E402
from mmdet3d.structures.ops import box_np_ops  # noqa: E402

AD_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = str(AD_ROOT / 'data' / 'kitti')
SAVE_DIR = AD_ROOT / 'data' / 'kitti'


def calculate_num_points_in_gt(data_path, infos, relative_path=True,
                               remove_outside=True, num_features=4):
    for info in infos:
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info['image_shape'])
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def main():
    imageset_folder = Path(DATA_PATH) / 'ImageSets'
    with open(imageset_folder / 'val.txt') as f:
        val_img_ids = [int(line) for line in f.readlines()]

    print(f'val frames: {len(val_img_ids)}')

    kitti_infos_val = kitti_data_utils.get_kitti_image_info(
        DATA_PATH,
        training=True,
        label_info=True,
        velodyne=True,
        calib=True,
        with_plane=False,
        image_ids=val_img_ids,
        relative_path=True)

    calculate_num_points_in_gt(DATA_PATH, kitti_infos_val, True)

    out = SAVE_DIR / 'kitti_infos_val.pkl'
    mmengine.dump(kitti_infos_val, out)
    print(f'saved to {out} ({len(kitti_infos_val)} frames)')


if __name__ == '__main__':
    main()
