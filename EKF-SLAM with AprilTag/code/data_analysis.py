import os
import numpy as np



# square length of square
SQ_LEN = 1.7

# landmark square length
LM_LEN = 2.4

# how many states for the car
STATE_SIZE = 3

# how many states for the landmarks
LM_SIZE = 2



def divide_line_into_three_equal_parts(point1, point2):
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    point_a = p1 + (p2 - p1) / 3
    point_b = p1 + 2 * (p2 - p1) / 3
    
    return tuple(point_a), tuple(point_b)


def generate_lm_coordinates():
    lm_len = LM_LEN
    sq_len = SQ_LEN

    left_bottom = ( 0 - ((lm_len - sq_len)/2) ,  0 - ((lm_len - sq_len)/2) )
    right_bottom = ( sq_len + ((lm_len - sq_len)/2) ,  0 - ((lm_len - sq_len)/2) )
    left_top = ( 0 - ((lm_len - sq_len)/2) , sq_len + ((lm_len - sq_len)/2) )
    right_top = ( sq_len + ((lm_len - sq_len)/2) , sq_len + ((lm_len - sq_len)/2) )

    top_point1, top_point2 = divide_line_into_three_equal_parts(left_top, right_top)
    left_point1, left_point2 = divide_line_into_three_equal_parts(left_top, left_bottom)
    bottom_point1, bottom_point2 = divide_line_into_three_equal_parts(left_bottom, right_bottom)
    right_point1, right_point2 = divide_line_into_three_equal_parts(right_top, right_bottom)

    correct_coordinates = {
        "marker_0": bottom_point1, 
        "marker_5": top_point1, 
        "marker_4": top_point2,
        "marker_7": left_point1,
        "marker_8": left_point2,
        "marker_1": bottom_point2,
        "marker_3": right_point1,
        "marker_2": right_point2,
    }

    return top_point1, top_point2, left_point1, left_point2, bottom_point1, bottom_point2, right_point1, right_point2, correct_coordinates


correct_coordinates = generate_lm_coordinates()[-1]

folder_path = "octagon1_backup1"


frame_id_to_observation_id = {}
with open(os.path.join(folder_path, "frame_id_to_observation_id.txt"), 'r') as file:

    # Read each line one by one
    for line in file:
        temp = line.split(":")
        frame_id_to_observation_id[temp[0]] = int(temp[1][1])


# load xEst 

xEst = np.loadtxt(os.path.join(folder_path, "xEst.txt"))

predicted_coordinates = {}

for frame_id in frame_id_to_observation_id:
    predicted_coordinates[frame_id] = (xEst[STATE_SIZE + LM_SIZE * frame_id_to_observation_id[frame_id]] , 
    xEst[STATE_SIZE + (LM_SIZE * frame_id_to_observation_id[frame_id]) + 1])


error_accu = 0
for frame_id in predicted_coordinates:
    print(frame_id)
    print(f"correct coordinates: {correct_coordinates[frame_id]}")    
    print(f"predicted coordinates: {predicted_coordinates[frame_id]}")
    error = np.hypot( correct_coordinates[frame_id][0] - predicted_coordinates[frame_id][0] , correct_coordinates[frame_id][1] - predicted_coordinates[frame_id][1])
    print(f"Error: {error}")
    print()
    error_accu += error

print(f"Average Error: {error_accu / len(correct_coordinates)}")