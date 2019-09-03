import cv2
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format

def get_label_name(labelmap, labels):
    num_labels = len(labelmap.item)
    label_names = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                label_names.append(labelmap.item[i].display_name.encode('utf-8'))
                break
    return label_names

def draw(image, predictions, labelmap_file):
    if labelmap_file:
        labelmap = caffe_pb2.LabelMap()
        file = open(labelmap_file, 'r')
        text_format.Merge(str(file.read()), labelmap)
        file.close()

    for prediction in predictions:
        x, y, w, h = prediction[0]
        score = prediction[1]
        cls = prediction[2]

        x1 = max(0, np.floor(x + 0.5).astype(int))
        y1 = max(0, np.floor(y + 0.5).astype(int))

        x2 = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        y2 = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (x1,y1), (x2,y2), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(get_label_name(labelmap, cls), score),
                    (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,0,255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(cls, score))
        print('box coordinate x, y, w, h: {0}'.format(prediction[0]))
    return image


def letterbox_image(image, w, h):
    # image => (h, w, c) rgb format
    img = image.copy()
    img = img / 255.
    
    ih = img.shape[0]
    iw = img.shape[1]
    scale = min(float(w) / iw, float(h) / ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.full((h, w, 3), 0.0, dtype=np.float32)
    paste_w = (w - nw) // 2
    paste_h = (h - nh) // 2

    new_image[paste_h:paste_h + nh, paste_w: paste_w + nw, :] = resized_img

    new_image = np.transpose(new_image, (2, 0, 1))

    return new_image
