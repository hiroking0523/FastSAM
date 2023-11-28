import os
import cv2
import torch
import numpy as np
import fastsam
import judgement_util 
import argparse
from fastsam import FastSAM, FastSAMPrompt 

try:
    import clip  # for linear_assignment
except (ImportError, AssertionError, AttributeError):
    from ultralytics.yolo.utils.checks import check_requirements

    check_requirements('git+https://github.com/openai/CLIP.git')  # required before installing lap from source
    import clip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    return parser.parse_args()


def main(args):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE:{}".format(DEVICE))

    FAST_SAM_CHECKPOINT = "./weights/FastSAM.pt"
    print("FAST_SAM_CHECKPOINT:{}".format(FAST_SAM_CHECKPOINT))
    model = fastsam.FastSAM(FAST_SAM_CHECKPOINT)

    cap = cv2.VideoCapture("./movie/person.mp4")
    if cap.isOpened() == False:
        print("cv2.VideoCapture() faild.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("FPS:{} WIDTH:{} HEIGHT:{}".format(fps, width, height))

    counter = 0

    # 保存する動画のファイル名とフォーマットを設定
    out_filename = './output/output_person_bbox_textpr.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4フォーマット
    out = cv2.VideoWriter(out_filename, fourcc, fps, (int(width), int(height)))

    while cap.isOpened():
        ret, frame = cap.read()
        counter += 1
        if ret == True:
            if counter % 2 == 0:
                everything_results = model(
                    frame,
                    device=DEVICE,
                    retina_masks=True,
                    imgsz=1024,
                    conf=0.7,
                    iou=0.9,
                )
                
                prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
                format_results = prompt_process._format_results(everything_results[0], 0)
                cropped_boxes, cropped_images, not_crop, filter_id, annotations = prompt_process._crop_image(format_results)
                clip_model, preprocess = clip.load('ViT-B/32', device=DEVICE)
                scores = prompt_process.retrieve(clip_model, preprocess, cropped_boxes, args.text_prompt, device=DEVICE)

                # # 0.1以上のスコアを持つbboxのインデックスを取得
                threshold = 0.05
                scores_cpu = scores.cpu()  # CUDAデバイス上のテンソルをCPUメモリにコピー

                # # スコアが0.1以上の要素を抽出し、そのインデックスを取得
                high_score_indices = torch.nonzero(scores_cpu > threshold).squeeze(dim=1).tolist()  # リストに変換

                # # スコアが高い順にインデックスをソート
                sorted_indices = sorted(high_score_indices, key=lambda idx: scores_cpu[idx], reverse=True)

                # bbox情報を取得し、bbox_tensorの最初の4つの要素を順番にリストに追加
                bbox_info_list = []
                for idx in sorted_indices:
                    max_idx = idx + sum(np.array(filter_id) <= int(idx))
                    bbox_tensor = annotations[max_idx]['bbox'].cpu()
                    bbox_info = bbox_tensor[:4].tolist()
                    bbox_info = [int(coord) for coord in bbox_info]  # 座標を整数に変換
                    bbox_info_list.append(bbox_info)

                try:
                    print("All annotations :{}".format(len(annotations)))
                except Exception as e:
                    print("EROR: {}".format(e))
                    continue 

                inference_frame = frame.copy()

                # すべてのバウンディングボックスを `inference_frame` に描画する
                for x, y, w, h in bbox_info_list:
                    cv2.rectangle(inference_frame, (x, y), (w, h), (0, 255, 0), 2)

                # 処理されたフレームを動画に書き込む
                out.write(inference_frame)
                # cv2.imshow("Org", frame)
                # cv2.imshow("Inference", inference_frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break
        else:
            break

    cap.release()
    out.release()  # VideoWriterを解放
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    main(args)

