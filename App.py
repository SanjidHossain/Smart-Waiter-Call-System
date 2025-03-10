import cv2
import mediapipe as mp
import numpy as np
import easyocr
import time

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])
name_results = reader.readtext('IMG.png')
# Convert name results into a dictionary of name: bbox
names = {}
for result in name_results:
    bbox = np.array(result[0], dtype=np.int32)
    text = result[1]
    # Extract bounding box as min/max
    x_coords = bbox[:,0]
    y_coords = bbox[:,1]
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    # Add some padding if needed (though we no longer show bounding boxes, we keep this for logic)
    padding = 20
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += 95  # as previously, though not strictly needed now
    names[text] = (x_min, y_min, x_max, y_max)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture('desk_video.mp4')

# Variables to control name display
current_name_displayed = None
display_start_time = None
display_duration = 2.0  # seconds

def is_point_in_box(x, y, box):
    x_min, y_min, x_max, y_max = box
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)

def box_overlap_score(arm_points, box):
    # Count how many arm landmark points fall inside the box
    count = 0
    for (px, py) in arm_points:
        if is_point_in_box(px, py, box):
            count += 1
    return count

def box_distance_to_arm(arm_points, box):
    # If no points inside, measure distance from arm's average point to box center
    x_min, y_min, x_max, y_max = box
    box_cx = (x_min + x_max) / 2.0
    box_cy = (y_min + y_max) / 2.0
    arm_cx = np.mean([p[0] for p in arm_points])
    arm_cy = np.mean([p[1] for p in arm_points])
    dist = np.sqrt((box_cx - arm_cx)**2 + (box_cy - arm_cy)**2)
    return dist

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        left_arm_raised = False
        right_arm_raised = False

        arm_points_left = []
        arm_points_right = []

        if results.pose_landmarks:
            # Extract keypoints
            landmarks = results.pose_landmarks.landmark
            # Convert normalized coords to pixel coords
            def to_pixel_coords(lm):
                return int(lm.x * w), int(lm.y * h)

            left_shoulder = to_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
            left_elbow = to_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW])
            left_wrist = to_pixel_coords(landmarks[mp_pose.PoseLandmark.LEFT_WRIST])

            right_shoulder = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            right_elbow = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW])
            right_wrist = to_pixel_coords(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])

            # Prepare arrays for convenience
            arm_points_left = [left_shoulder, left_elbow, left_wrist]
            arm_points_right = [right_shoulder, right_elbow, right_wrist]

            # Check if left arm is raised
            if (left_wrist[1] < left_shoulder[1]) and (left_elbow[1] < left_shoulder[1]):
                left_arm_raised = True

            # Check if right arm is raised
            if (right_wrist[1] < right_shoulder[1]) and (right_elbow[1] < right_shoulder[1]):
                right_arm_raised = True

            # Additional checks from original code
            if (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y - landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y) > 0.01:
                left_arm_raised = True
            if (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y - landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y) > 0.01:
                right_arm_raised = True

        # If an arm is raised, determine whose arm it is
        arm_just_raised = (left_arm_raised or right_arm_raised)
        if arm_just_raised:
            # Determine which name box corresponds to the raised arm
            chosen_arm_points = arm_points_left if left_arm_raised else arm_points_right

            # Find the best matching box
            best_name = None
            best_score = -1
            best_distance = float('inf')

            for name, box in names.items():
                score = box_overlap_score(chosen_arm_points, box)
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_distance = box_distance_to_arm(chosen_arm_points, box)
                elif score == best_score and score == 0:
                    # If tie in score and score=0 (no overlap), pick closer one
                    dist = box_distance_to_arm(chosen_arm_points, box)
                    if dist < best_distance:
                        best_distance = dist
                        best_name = name

            # Show the chosen name for 2 seconds if different from currently displayed or if display expired
            current_time = time.time()
            if best_name:
                if current_name_displayed is None or (current_time - display_start_time > display_duration):
                    current_name_displayed = best_name
                    display_start_time = current_time

        # Handle the timing for current_name_displayed
        if current_name_displayed is not None:
            current_time = time.time()
            if current_time - display_start_time <= display_duration:
                # Still within display window
                # Display the name in the top-right corner
                # Calculate a position near the top-right corner
                text = current_name_displayed
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_width, text_height = text_size
                x_pos = w - text_width - 20
                y_pos = 50  # 50px down from the top

                cv2.putText(frame, text, (x_pos, y_pos), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

            else:
                # Time exceeded 2 seconds, remove the name
                current_name_displayed = None

        # We will still show "Arm Raised" text if desired. If you prefer not, comment out below lines.
        if left_arm_raised:
            cv2.putText(frame, "Left Arm Raised", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if right_arm_raised:
            cv2.putText(frame, "Right Arm Raised", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Arm Raised Detection with Names', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()