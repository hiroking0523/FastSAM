import os
import cv2
import torch
import numpy as np
import fastsam
import judgement_util 


def collision_detection(line, rect):
    result = judgement_util.line_polygon_intersection(line[0], line[1], rect)
    return len(result) > 0


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

    # 道路のライン
    # y0 = 260  # 道路の上限
    # y1 = 600  # 道路の下限
    # up_0 = [(280, y0), (120, y1)]  # 上り左車線
    # up_1 = [(310, y0), (240, y1)]  # 上り右車線（追い越し）
    # down_0 = [(405, y0), (605, y1)]  # 下り左車線
    # down_1 = [(375, y0), (485, y1)]  # 下り右車線（追い越し）

    # add by nishi（高速道路用）
    # down_0 = [(600, y0), (1080, y1)]  # 下り左車線
    # down_1 = [(335, y0), (670, y1)]  # 下り右車線（追い越し）

    # add by nishi（person用）
    center = (497, 268)
    down_0 = [(13, 204), center]  # 左
    down_1 = [(760, 409), center]  # 右
    down_2 = [(750, 110), center]  # 上

    counter = 0

    # 保存する動画のファイル名とフォーマットを設定
    out_filename = './output/output_person_bbox.mp4'
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
                contour_up_0 = []
                contour_up_1 = []
                contour_down_0 = []
                contour_down_1 = []
                bbox_down_0 = []  # down_0 のバウンディングボックスを保存するリスト
                bbox_down_1 = []  # down_1 のバウンディングボックスを保存するリスト
                bbox_down_2 = []  # down_2 のバウンディングボックスを保存するリスト
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
                        rect = [(x, y), (x, y + h), (x + w, y + h), (x + w, y)]
                        # 道路ラインとの衝突検知（セグメンテーション用）
                        # if collision_detection(up_0, rect):
                        #     contour_up_0.append(contour)
                        # if collision_detection(up_1, rect):
                        #     contour_up_1.append(contour)
                        # if collision_detection(down_0, rect):
                        #     contour_down_0.append(contour)
                        # if collision_detection(down_1, rect):
                        #     contour_down_1.append(contour)

                        # add by nishi（bbox用）
                        if collision_detection(down_0, rect):
                            bbox_down_0.append((x, y, w, h))
                        if collision_detection(down_1, rect):
                            bbox_down_1.append((x, y, w, h))
                        if collision_detection(down_2, rect):
                            bbox_down_2.append((x, y, w, h))

                # セグメンテーションを `inference_frame` に描画する
                # cv2.drawContours(inference_frame, contour_up_0, -1, (255, 150, 0), 2)
                # cv2.drawContours(inference_frame, contour_up_1, -1, (255, 0, 0), 2)
                # cv2.drawContours(inference_frame, contour_down_0, -1, (0, 150, 255), 2)
                # cv2.drawContours(inference_frame, contour_down_1, -1, (0, 0, 255), 2)

                # バウンディングボックスを `inference_frame` に描画する
                for x, y, w, h in bbox_down_0:
                    cv2.rectangle(inference_frame, (x, y), (x + w, y + h), (0, 150, 255), 2)
                for x, y, w, h in bbox_down_1:
                    cv2.rectangle(inference_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                for x, y, w, h in bbox_down_2:
                    cv2.rectangle(inference_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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


main()