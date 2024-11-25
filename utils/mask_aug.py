import cv2
import bezier
import random
import numpy as np


def extend_mask_with_bezier(mask, extend_ratio=0.2, random_width=5):

    H, W = mask.shape

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    extended_mask = np.zeros((H, W), dtype=np.uint8)

    for contour in contours:
        bbox = cv2.boundingRect(contour)
        x, y, w, h = bbox

        extended_bbox = [
            x - int(extend_ratio * w),
            y - int(extend_ratio * h),
            x + w + int(extend_ratio * w),
            y + h + int(extend_ratio * h),
        ]

        extended_bbox[0] = max(0, extended_bbox[0])
        extended_bbox[1] = max(0, extended_bbox[1])
        extended_bbox[2] = min(W, extended_bbox[2])
        extended_bbox[3] = min(H, extended_bbox[3])

        top_nodes = np.asfortranarray(
            [[x, (x + x + w) // 2, x + w], [y, extended_bbox[1], y]]
        )
        down_nodes = np.asfortranarray(
            [[x + w, (x + x + w) // 2, x], [y + h, extended_bbox[3], y + h]]
        )
        left_nodes = np.asfortranarray(
            [[x, extended_bbox[0], x], [y + h, (y + y + h) // 2, y]]
        )
        right_nodes = np.asfortranarray(
            [[x + w, extended_bbox[2], x + w], [y, (y + y + h) // 2, y + h]]
        )

        top_curve = bezier.Curve(top_nodes, degree=2)
        right_curve = bezier.Curve(right_nodes, degree=2)
        down_curve = bezier.Curve(down_nodes, degree=2)
        left_curve = bezier.Curve(left_nodes, degree=2)

        pt_list = []
        for curve in [top_curve, right_curve, down_curve, left_curve]:
            for i in range(1, 20):
                pt = curve.evaluate(i * 0.05)
                pt_list.append(
                    (
                        int(pt[0, 0] + random.randint(-random_width, random_width)),
                        int(pt[1, 0] + random.randint(-random_width, random_width)),
                    )
                )
        cv2.fillPoly(extended_mask, [np.array(pt_list)], 1)

    return extended_mask * 255


def mask_paint2bbox(mask, random_drop=0.0):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x, y, w, h = cv2.boundingRect(contours[0])
    new_mask = np.zeros_like(mask)
    # if random_drop > 0 and random.random() < random_drop:
    #     w = w * (random.random() + 0.5)
    #     h = h * (random.random() + 0.5)
        
    cv2.rectangle(new_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return new_mask
