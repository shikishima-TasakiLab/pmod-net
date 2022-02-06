import argparse
from ..model.constant import Q_IDX
import os
from typing import Dict, Tuple
from h5dataloader.common.structure import *
import numpy as np
import h5py
from tqdm import tqdm
import cv2

TAG_HEIGHT = 32
PADDING = 4
FONT_SCALE = 0.8
FONT_COLOR = (255, 255, 255)
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 2


def parse_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input',
        type=str, metavar='PATH', required=True,
        help='Input path.'
    )
    parser.add_argument(
        '-o', '--output',
        type=str, metavar='PATH', default=None,
        help='Output path. Default is "[input dir]/data.avi"'
    )
    parser.add_argument(
        '-r', '--raw',
        action='store_true',
        help='Use raw codec'
    )
    args = vars(parser.parse_args())

    if isinstance(args['output'], str):
        if os.path.isdir(os.path.dirname(args['output'])) is False:
            raise NotADirectoryError(os.path.dirname(args['output']))
    else:
        output_dir: str = os.path.dirname(args['input'])
        args['output'] = os.path.join(output_dir, 'data.avi')
    return args


def main(args: Dict[str, str]):
    with h5py.File(args['input'], mode='r') as h5file:
        data_group: h5py.Group = h5file['data']
        label_group: h5py.Group = h5file['label']
        img_shape: Tuple[int, ...] = data_group['0/Input-Camera'].shape

        tag_bg = np.zeros((TAG_HEIGHT, img_shape[1], 3), dtype=np.uint8)
        pose_err_bg = np.zeros(
            (TAG_HEIGHT, img_shape[1] * 3, 3), dtype=np.uint8)

        _, text_baseline = cv2.getTextSize(
            'Input: Camera', FONT_FACE, FONT_SCALE, FONT_THICKNESS)
        text_x = PADDING
        text_y: int = TAG_HEIGHT - text_baseline - PADDING

        tag_in_camera: np.ndarray = np.copy(tag_bg)
        tag_in_map: np.ndarray = np.copy(tag_bg)
        tag_pr_seg: np.ndarray = np.copy(tag_bg)
        tag_pr_depth: np.ndarray = np.copy(tag_bg)
        tag_gt_seg: np.ndarray = np.copy(tag_bg)
        tag_gt_depth: np.ndarray = np.copy(tag_bg)

        cv2.putText(tag_in_camera,  'Input: Camera',        (text_x,
                    text_y), FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        cv2.putText(tag_in_map,     'Input: Map',           (text_x,
                    text_y), FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        cv2.putText(tag_pr_seg,     'Pred.: Segmentation',  (text_x,
                    text_y), FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        cv2.putText(tag_pr_depth,   'Pred.: Depth',         (text_x,
                    text_y), FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        cv2.putText(tag_gt_seg,     'GT: Segmentation',     (text_x,
                    text_y), FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        cv2.putText(tag_gt_depth,   'GT: Depth',            (text_x,
                    text_y), FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

        fourcc = 0 if args['raw'] is True else cv2.VideoWriter_fourcc(*'XVID')
        video_out = cv2.VideoWriter(
            args['output'], fourcc, 10.0, (img_shape[1] * 3, img_shape[0] * 2 + TAG_HEIGHT * 3))

        def convert_label(src: h5py.Dataset) -> np.ndarray:
            label_tag: str = src.attrs[H5_ATTR_LABELTAG]
            labels: h5py.Group = label_group[label_tag]

            color_label: np.ndarray = np.zeros(
                (src.shape[0], src.shape[1], 3), dtype=np.uint8)

            for label_key, label_values in labels.items():
                color_label[np.where(src[()] == int(label_key))
                            ] = label_values[TYPE_COLOR][()]
            return color_label

        def quat_inv(q: np.ndarray) -> np.ndarray:
            dst: np.ndarray = np.copy(q)
            dst[:3] *= -1.0
            return dst

        def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
            q: np.ndarray = np.zeros_like(q1)
            q[Q_IDX.w] = q2[Q_IDX.w] * q1[Q_IDX.w] - q2[Q_IDX.x] * \
                q1[Q_IDX.x] - q2[Q_IDX.y] * \
                q1[Q_IDX.y] - q2[Q_IDX.z] * q1[Q_IDX.z]
            q[Q_IDX.x] = q2[Q_IDX.w] * q1[Q_IDX.x] + q2[Q_IDX.x] * \
                q1[Q_IDX.w] - q2[Q_IDX.y] * \
                q1[Q_IDX.z] + q2[Q_IDX.z] * q1[Q_IDX.y]
            q[Q_IDX.y] = q2[Q_IDX.w] * q1[Q_IDX.y] + q2[Q_IDX.x] * \
                q1[Q_IDX.z] + q2[Q_IDX.y] * \
                q1[Q_IDX.w] - q2[Q_IDX.z] * q1[Q_IDX.x]
            q[Q_IDX.z] = q2[Q_IDX.w] * q1[Q_IDX.z] - q2[Q_IDX.x] * \
                q1[Q_IDX.y] + q2[Q_IDX.y] * \
                q1[Q_IDX.x] + q2[Q_IDX.z] * q1[Q_IDX.w]
            return q

        def depth2colormap(src: np.ndarray, range_min: float, range_max: float) -> np.ndarray:
            out_range = np.where((src < range_min) | (range_max < src))
            src_norm = np.uint8(
                (1.0 - (src - range_min) / (range_max - range_min)) * 255.0)
            colormap = cv2.applyColorMap(src_norm, cv2.COLORMAP_JET)
            colormap[out_range] = [0, 0, 0]
            return colormap

        for itr in tqdm(range(h5file['header/length'][()]), desc='HDF5 -> AVI'):
            src_group: h5py.Group = data_group[str(itr)]

            in_camera: np.ndarray = src_group['Input-Camera'][()]
            in_map: np.ndarray = depth2colormap(
                src_group['Input-Map'][()], 0.0, 100.0)
            pr_seg: np.ndarray = convert_label(src_group['Pred-Label'])
            pr_depth: np.ndarray = depth2colormap(
                src_group['Pred-Depth'][()], 0.0, 100.0)
            gt_seg: np.ndarray = convert_label(src_group['GT-Label'])
            gt_depth: np.ndarray = depth2colormap(
                src_group['GT-Depth'][()], 0.0, 100.0)

            pose_err = np.copy(pose_err_bg)
            if SUBTYPE_TRANSLATION in src_group.keys():
                pr_translation: np.ndarray = src_group['Pred-Pose'][SUBTYPE_TRANSLATION][()]
                pr_quaternion: np.ndarray = src_group['Pred-Pose'][SUBTYPE_ROTATION][()]

                gt_translation: np.ndarray = src_group['GT-Pose'][SUBTYPE_TRANSLATION][()]
                gt_quaternion: np.ndarray = src_group['GT-Pose'][SUBTYPE_ROTATION][()]

                err_translation = pr_translation - gt_translation
                err_quaternion = quat_mul(
                    gt_quaternion, quat_inv(pr_quaternion))
                err_rot = np.degrees(np.arctan2(np.linalg.norm(
                    err_quaternion[:3], ord=2), np.abs(err_quaternion[Q_IDX.w])) * 2.0)

                cv2.putText(pose_err, f'Translation Error: ({err_translation[0]:.4f}, {err_translation[1]:.4f}, {err_translation[2]:.4f}) [m], Rotation Error: {err_rot:.2f} [deg]', (
                    text_x, text_y), FONT_FACE, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

            in_img = np.vstack([tag_in_camera, in_camera, tag_in_map, in_map])
            pr_img = np.vstack([tag_pr_seg, pr_seg, tag_pr_depth, pr_depth])
            gt_img = np.vstack([tag_gt_seg, gt_seg, tag_gt_depth, gt_depth])
            img = np.hstack([in_img, pr_img, gt_img])
            view = np.vstack([img, pose_err])

            video_out.write(view)

        video_out.release()
        print(f'Saved: {args["output"]}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
