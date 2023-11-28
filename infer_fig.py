import cv2
import torch
import numpy as np
import fastsam
import judgement_util 


def collision_detection(line, rect):
    result = judgement_util.line_polygon_intersection(line[0], line[1], rect)
    return len(result) > 0


def image_inference(image_path):
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE:{}".format(DEVICE))

    FAST_SAM_CHECKPOINT = "./weights/FastSAM.pt"
    print("FAST_SAM_CHECKPOINT:{}".format(FAST_SAM_CHECKPOINT))
    model = fastsam.FastSAM(FAST_SAM_CHECKPOINT)

    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read image from {}".format(image_path))
        return

    # Define lines for collision detection here if needed

    everything_results = model(
        image,
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
        print("ERROR: {}".format(e))
        return

    annotations = annotations.cpu().numpy()

    inference_image = image.copy()
    # Add contour and bbox lists here if needed

    for mask in annotations:
        annotation = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Add collision detection and bounding box drawing code here if needed

    # Add code to display or save the inference image here if needed

    cv2.imshow("Inference", inference_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with the path to your image
image_inference("./images/dogs.jpg")
