class frame_data_class:
    def __init__(self, frame_id, frame_l, frame_r, boxes, masks, classes):
        self.frame_id = frame_id
        self.frame_l = frame_l
        self.frame_r = frame_r
        self.boxes = boxes
        self.masks = masks
        self.classes = classes
        self.ids = []
        self.max_id = None
        self.positions_3d = []
        self.velocities = []
        self.radii = []