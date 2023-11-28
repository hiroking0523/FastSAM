import os
import cv2
import torch
import numpy as np
import fastsam
import judgement_util 


def main():
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
    out_filename = './output/output_person_bbox_all.mp4'
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
                    conf=0.8,
                    iou=0.9,
                )
                annotations = everything_results[0].masks.data
                try:
                    print("All annotations :{}".format(len(annotations)))
                except Exception as e:
                    print("EROR: {}".format(e))
                    continue

                annotations = annotations.cpu().numpy()

                inference_frame = frame.copy()
                bbox_all = []  # すべてのバウンディングボックスを保存するリスト
                for i, mask in enumerate(annotations):
                    annotation = mask.astype(np.uint8)
                    contours, hierarchy = cv2.findContours(
                        annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for contour in contours:
                        # 外接の矩形を求める
                        x, y, w, h = cv2.boundingRect(contour)
                        if w >=60 & h >= 160:  # 車のサイズとして異常値を排除する（車線等が検出されてしまうため）
                            continue
                        bbox_all.append((x, y, w, h))

                # すべてのバウンディングボックスを `inference_frame` に描画する
                for x, y, w, h in bbox_all:
                    cv2.rectangle(inference_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 処理されたフレームを動画に書き込む
                out.write(inference_frame)
                cv2.imshow("Org", frame)
                cv2.imshow("Inference", inference_frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

        else:
            break

    cap.release()
    out.release()  # VideoWriterを解放
    cv2.destroyAllWindows()


main()
