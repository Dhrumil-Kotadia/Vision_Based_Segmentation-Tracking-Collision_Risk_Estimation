import cv2
import os
import numpy as np
from copy import deepcopy

from utils.im_utils import *
from utils.segmentation_utils import *
from utils.datatype_utils import *
from utils.tracking_utils import *


if __name__ == "__main__":

    image_directory_l = "data/l/"
    image_directory_r = "data/r/"

    images_l = Load_Images(image_directory_l)
    images_r = Load_Images(image_directory_r)
    print("Number of images:", len(images_l))

    output_directory = "output/"
    os.makedirs(output_directory, exist_ok=True)

    frame_data = []
    global_translation = np.zeros((3, 1))
    global_rotation = np.eye(3)

    k_matrix = np.array([[707.0912, 0, 601.8873],
        [0, 707.0912, 183.1104],
        [0, 0, 1]])

    for image_index, image_l in enumerate(images_l):
        print("processing Frame: ", image_index)
        
        frame_l = images_l[image_index]
        frame_r = images_r[image_index]

        features1, descriptors1 = Extract_Features(images_l[image_index])
        features2, descriptors2 = Extract_Features(images_l[image_index+1])

        matches = Match_Features(descriptors1, descriptors2)

        r, t, mask = estimate_pose(features1, features2, matches, k_matrix)

        Points3D = triangulate_points(features1, features2, matches, global_rotation, global_translation, r, t, k_matrix, mask)

        if len(Points3D) == 0:
            continue
        for Iter in range(len(Points3D)):
            UpdatedPoints3D = Points3D[Iter] + global_translation

        global_translation = global_translation + np.dot(global_rotation, t)
        global_rotation = np.dot(r, global_rotation)
        frame = deepcopy(frame_l)
        frame, boxes, masks, classes = process_frame(frame)
        data = frame_data_class(image_index, frame_l, frame_r, boxes, masks, classes)

        if image_index == 0 or frame_data[-1].max_id is None:
            data.ids = list(range(len(boxes)))
            if len(boxes) > 0:
                data.max_id = len(boxes)
        elif frame_data[-1].max_id is not None:
            data.ids = [(frame_data[-1].max_id + 1) for i in range(len(boxes))] 
            update_ids(frame_data[-1], data)
        
        depth_map = get_depths(data)
        for i in range(len(data.boxes)):
            mask = masks[i]
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            cv2.putText(frame_l, str(data.ids[i]), (int(data.boxes[i][0]), int(data.boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            curr_position = (int((data.boxes[i][0]+data.boxes[i][2])/2), int(data.boxes[i][3]))
            curr_position_3d = backproject_2d_to_3d(curr_position, k_matrix, global_rotation, global_translation, depth_map[curr_position[1], curr_position[0]])
            data.positions_3d.append(curr_position_3d)
            
            if len(frame_data[-1].ids) > i:
                prev_position = frame_data[-1].positions_3d[i]
                prev_velocity = frame_data[-1].velocities[i]
                curr_velocity = (curr_position_3d - prev_position)/0.1
                
                if np.linalg.norm(curr_velocity) > 8:
                    curr_velocity = 0.9*prev_velocity + 0.1*curr_velocity
                else:
                    curr_velocity = 0.7*prev_velocity + 0.3*curr_velocity
                data.velocities.append(curr_velocity)
                
                next_position_3d = curr_position_3d + (curr_velocity*0.1)
                next_position_pred = project_3d_to_2d(next_position_3d, k_matrix, global_rotation, global_translation)
                next_position_pred = (int(next_position_pred[0]), int(next_position_pred[1]))
                
                radius = int(np.linalg.norm(curr_velocity)*0.2)
                radius = 5 + (radius)
                
                danger_zone = create_danger_zone(curr_position_3d, radius, k_matrix, global_rotation, global_translation)
                danger_zone = np.array(danger_zone)
                
                cv2.fillPoly(frame, [danger_zone], (0, 0, 255))
                for pt in danger_zone:
                    cv2.circle(frame_l, tuple(pt), 0, (0, 0, 255), -1)            
            
            else:    
                data.velocities.append(np.zeros((3, 1)))
                data.radii.append(5)
            
            frame = np.where(mask, [0, 0, 255], frame).astype(np.uint8)
            
        frame_data.append(data)
        out_image = cv2.addWeighted(frame_l, 0.8, frame, 0.2, 0)
            
        cv2.imwrite(f"{output_directory}output_{image_index}.png", out_image)
        


        
