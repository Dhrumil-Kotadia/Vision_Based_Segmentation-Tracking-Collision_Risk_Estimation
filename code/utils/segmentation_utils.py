import torch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

def initialize_maskrcnn():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    predictor = DefaultPredictor(cfg)

    return predictor


def process_frame(frame):
    predictor = initialize_maskrcnn()
    outputs = predictor(frame)
    
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()

    indices = [i for i, c in enumerate(classes) if c == 0]
    boxes, scores, masks, classes = boxes[indices], scores[indices], masks[indices], classes[indices]

    indices = scores > 0.8
    boxes, scores, masks, classes = boxes[indices], scores[indices], masks[indices], classes[indices]

    indices = [i for i, b in enumerate(boxes) if (b[2] - b[0]) * (b[3] - b[1]) > 800]
    boxes, scores, masks, classes = boxes[indices], scores[indices], masks[indices], classes[indices]

    return frame, boxes, masks, classes