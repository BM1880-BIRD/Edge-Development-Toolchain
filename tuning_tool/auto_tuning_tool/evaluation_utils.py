import numpy as np

def L2_diff(net8_out, net32_out):
    return (np.square(net32_out - net8_out)).mean(axis=None)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x, t=-100.):
        x = x - np.max(x)
        if np.min(x) < t:
            x = x / np.min(x) * t
        exp_x = np.exp(x)
        out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return out

def process_feats(out, anchors):
        grid_size = out.shape[2]
        num_boxes_per_cell = 3

        out = np.transpose(out, (1, 2, 0))
        out = np.reshape(out, (grid_size, grid_size, num_boxes_per_cell, out.shape[-1]/num_boxes_per_cell))

        threshold_predictions = []

        anchors_tensor = np.array(anchors).reshape(1, 1, 3, 2)

        box_xy = sigmoid(out[..., :2])
        box_wh = np.exp(out[..., 2:4]) * anchors_tensor

        box_confidence = sigmoid(out[..., 4])
        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = softmax(out[..., 5:])

        col = np.tile(np.arange(0, grid_size), grid_size).reshape(-1, grid_size)
        row = np.tile(np.arange(0, grid_size).reshape(-1, 1), grid_size)

        col = col.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_size, grid_size, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_size, grid_size)

        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        box_score = box_confidence * box_class_probs
        box_classes = np.argmax(box_score, axis=-1)
        box_class_score = np.max(box_score, axis=-1)

        scores = np.expand_dims(box_class_score, axis=-1)
        classes = np.expand_dims(box_classes, axis=-1)

        if boxes is not None:
            threshold_predictions = np.concatenate((boxes, scores, classes), axis=-1)

        return threshold_predictions

def yolo_diff(net8_out, net32_out):
    feat8 = [net8_out['layer82-conv'][0], net8_out['layer94-conv'][0], net8_out['layer106-conv'][0]]
    feat32 = [net32_out['layer82-conv'][0], net32_out['layer94-conv'][0], net32_out['layer106-conv'][0]]

    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    total = 0
    for i, out in enumerate(feat8):
        predictions8 = process_feats(feat8[i], anchors[i])
        predictions32 = process_feats(feat32[i], anchors[i])

        total += L2_diff(predictions8, predictions32)

    return total