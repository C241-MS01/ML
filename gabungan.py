import cv2
import time
import mediapipe as mp
import numpy as np
import os
import torch



def combined_detection(model_path, folder_path, object_labels, fps):
    # Process frames for face detection and driver behavior analysis
    def mediapipe_detection(folder_path):
        # --- Drawing and Create Face Mesh on Face ---
        mpDraw = mp.solutions.drawing_utils
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        denormalize_coordinates = mpDraw._normalized_to_pixel_coordinates
        drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

        # --- Landmark of eye ---
        # landmark points to left eye
        all_left_eye_idxs = list(mpFaceMesh.FACEMESH_LEFT_EYE)
        # flatten and remove duplicates
        all_left_eye_idxs = set(np.ravel(all_left_eye_idxs))

        # landmark points to right eye
        all_right_eye_idxs = list(mpFaceMesh.FACEMESH_RIGHT_EYE)
        # flatten and remove duplicates
        all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))

        # landmark points to lips
        lips_idxs = list(mpFaceMesh.FACEMESH_LIPS)

        # Combined for plotting Landmark points for both eye
        all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)

        # The chosen 12 (left and right) points: P1 - P6
        chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
        chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]
        all_eyes_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

        # lips chosen 8 points mouth: P1 - P8
        lips_idxs = [61, 39, 0, 269, 291, 405, 17, 181]

        # nose chosen 6 points mouth: P1 - P6
        nose_idxs = [33, 263, 1, 61, 291, 199]

        # combine for all chosen idxs
        all_chosen_idxs = list(set(all_eyes_idxs + lips_idxs + nose_idxs))

        # --- Formula ---
        # distance for eye and lips
        def distance(point_1, point_2):
            dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
            return dist

        # --- Formula Eye Aspect Ratio (EAR) ---
        # Calculate 12-norm between two points
        # Calculate EAR for one eye
        def get_ear(landmarks, refer_idxs, frame_width, frame_height):
            try:
                coords_points = []
                for i in refer_idxs:
                    lm = landmarks[i]
                    coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
                    coords_points.append(coord)
                
                P2_P6 = distance(coords_points[1], coords_points[5])
                P3_P5 = distance(coords_points[2], coords_points[4])
                P1_P4 = distance(coords_points[0], coords_points[3])
            
                ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

            except:
                ear = 0.0
            coords_points = None

            return ear, coords_points

        # Calculate EAR
        def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h): 
            left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
            right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
            Avg_EAR = (left_ear + right_ear) / 2.0
            return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

        # --- Formula Mouth Aspect Ratio (MAR) ---
        # Calculate MAR
        def calculate_mar(landmarks, refer_idxs, image_w, image_h):
            try:
                coords_points = []
                for i in refer_idxs:
                    lm = landmarks[i]
                    coord = denormalize_coordinates(lm.x, lm.y, image_w, image_h)
                    coords_points.append(coord)
                
                P2_P8 = distance(coords_points[1], coords_points[7])
                P3_P7 = distance(coords_points[2], coords_points[6])
                P4_P6 = distance(coords_points[3], coords_points[5])
                P1_P5 = distance(coords_points[0], coords_points[4])
            
                mar = (P2_P8 + P3_P7 + P4_P6) / (2.0 * P1_P5)

            except:
                mar = 0.0
            coords_points = None

            return mar, coords_points

        # --- Detection Function ---
        # Sleeping detection
        def detect_sleeping(EAR, ear_thresh, ear_time_thresh, img):
            global ear_below_thresh_time, start_time
            sleep_duration = 0
            sleep_condition = False

            if EAR < ear_thresh:
                if ear_below_thresh_time == 0:
                    start_time = time.perf_counter()
                ear_below_thresh_time = time.perf_counter() - start_time
                                
                cv2.putText(img, 
                            text="Close Eyes", 
                            org=(15, 35), 
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1, 
                            color=RED,
                            thickness=2, 
                            lineType=cv2.LINE_AA)

                if ear_below_thresh_time >= ear_time_thresh:
                    sleep_duration = ear_below_thresh_time
                    sleep_condition = True
                    cv2.putText(img, 
                                text="Driver is sleeping!", 
                                org=(15, 120), 
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=1, 
                                color=(0, 0, 255),
                                thickness=2, 
                                lineType=cv2.LINE_AA)
            else:
                ear_below_thresh_time = 0

            return sleep_duration, sleep_condition

        def detect_yawning(MAR, mar_thresh, mar_time_thresh, img):
            global mar_below_thresh_time, start_time
            yawn_duration = 0
            yawn_condition = False

            if MAR > mar_thresh:
                if mar_below_thresh_time == 0:
                    start_time = time.perf_counter()
                mar_below_thresh_time = time.perf_counter() - start_time
                                
                cv2.putText(img, 
                            text="Open mouth", 
                            org=(15, 35), 
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1, 
                            color=RED,
                            thickness=2, 
                            lineType=cv2.LINE_AA)

                if mar_below_thresh_time >= mar_time_thresh:
                    yawn_duration = mar_below_thresh_time
                    yawn_condition = True
                    cv2.putText(img, 
                                text="Driver is yawning!", 
                                org=(15, 120), 
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=1, 
                                color=(0, 0, 255),
                                thickness=2, 
                                lineType=cv2.LINE_AA)
            else:
                mar_below_thresh_time = 0

            return yawn_duration, yawn_condition

        def detect_head_focus(landmarks, img_w, img_h, focus_time_thresh):
            global focus_below_thresh_time, start_time, notfocus_duration, notfocus_condition
            notfocus_duration = 0
            notfocus_condition = False

            try:
                face_2d = []
                face_3d = []
                
                for idx in nose_idxs:
                    lm = landmarks[idx]
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                
                # Get 2d coord
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                # Cam Matrix
                cam_matrix = np.array([[focal_length, 0, img_w / 2], 
                                    [0, focal_length, img_h / 2], 
                                    [0, 0, 1]])
                
                # Distortion Matrix
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                # rmat
                rmat, _ = cv2.Rodrigues(rotation_vec)
                
                # Getting Angles
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360

                # Calculated Axis rot angle
                if y < -10:
                    text = "Looking Right"
                elif y > 10:
                    text = "Looking Left"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                if text != "Forward":
                    if focus_below_thresh_time == 0:
                        start_time = time.perf_counter()
                    focus_below_thresh_time = time.perf_counter() - start_time

                    if focus_below_thresh_time >= focus_time_thresh:
                        notfocus_duration = focus_below_thresh_time
                        notfocus_condition = True
                        cv2.putText(img, 
                                text="Driver not Focus!", 
                                org=(15, 120), 
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=1, 
                                color=(0, 0, 255),
                                thickness=2, 
                                lineType=cv2.LINE_AA)
                else:
                    focus_below_thresh_time = 0

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                return text, p1, p2, x, y, notfocus_duration, notfocus_condition 
            
            except Exception as e:
                print(f"Error: {e}")
                return "Forward", (0, 0), (0, 0), 0, 0, notfocus_duration, notfocus_condition

        # --- info before start ---
        # image resize
        width = 800
        height = 450

        # color code
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)

        # key dictionary
        dect_class = ['ear', 'mar', 'sleep_duration', 'yawning_duration']

        # declare and time status
        sleep_condition = False
        yawn_condition = False
        notfocus_condition = False
        sleep_duration = 0
        yawn_duration = 0
        notfocus_duration = 0

        # threshold for detection
        ear_thresh = 0.13
        mar_thresh = 1.0
        ear_time_thresh = 1
        mar_time_thresh = 1
        focus_time_thresh = 1
        ear_below_thresh_time = 0
        mar_below_thresh_time = 0
        focus_below_thresh_time = 0
        start_time = 0

        # Dictionary to store results
        detection_results = {}

        # --- start video detection ---
        frame_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

        for frame_file in frame_files:
            frame_path = os.path.join(folder_path, frame_file)
            img = cv2.imread(frame_path)

            if img is None:
                continue

            # resize the video
            img = cv2.resize(img, (width, height))

            # Convert the BGR to RGB image
            img.flags.writeable = False
            img_h, img_w, img_c = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = faceMesh.process(img)

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:    
                landmarks = results.multi_face_landmarks[0].landmark
                for faceLms in results.multi_face_landmarks:
                    # EAR and MAR Calculate 
                    EAR, coordinates = calculate_avg_ear(landmarks, chosen_left_eye_idxs, chosen_right_eye_idxs, img_w, img_h)
                    MAR, coordinates = calculate_mar(landmarks, lips_idxs, img_w, img_h)

                    sleep_duration, sleep_condition = detect_sleeping(EAR, ear_thresh, ear_time_thresh, img)
                    yawn_duration, yawn_condition = detect_yawning(MAR, mar_thresh, mar_time_thresh, img)
                    
                    head_pose_text, p1, p2, x, y, notfocus_duration, notfocus_condition = detect_head_focus(landmarks, img_w, img_h, focus_time_thresh)

                    cv2.line(img, p1, p2, (255, 0, 0), 3)
                    cv2.putText(img, head_pose_text, (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"x: {np.round(x, 2)}", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(img, f"y: {np.round(y, 2)}", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if sleep_condition == False and yawn_condition == False and notfocus_condition == False:
                        cv2.circle(img, (15,15), 2, GREEN, -1)
                        cv2.putText(img, "Steady", (15, 35), cv2.FONT_HERSHEY_DUPLEX, 1, GREEN, 2)
                    
                    else:
                        data = {
                            "EAR": EAR,
                            "MAR": MAR,
                            "Sleep Duration": sleep_duration,
                            "Yawning Duration": yawn_duration,
                            "Focus Duration": notfocus_duration
                        }
                        print("\n")
                        for key, value in enumerate(data.items()):
                            print(f"{key}: {value}")
                        
                        # Add data to detection_results
                        detection_results[frame_file] = data

            fps = int(1 / (time.perf_counter() - start_time))
            cv2.putText(img, f'FPS: {fps}', (15, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

            #cv2.imshow('Head Pose, Yawning and Sleep Detection', img)
            #cv2.waitKey(1)

        cv2.destroyAllWindows()
        return detection_results

    
    # Detect objects using YOLOv5
    def detect_objects_from_folder(model_path, folder_path, object_labels, fps):
        # Load frames from the folder in one line
        frames = [cv2.imread(os.path.join(folder_path, filename)) for filename in sorted(os.listdir(folder_path)) if filename.endswith(('.png', '.jpg', '.jpeg'))]

        # Load YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

        # Variables to track detected object and its duration
        detected_object = None
        start_time = None
        total_duration = {label: 0 for label in object_labels}

        processed_frames = []

        # Process each frame
        for frame_count, frame in enumerate(frames):
            # Perform object detection on the frame
            results = model(frame)

            # Flags to indicate if any object is detected in the current frame
            object_detected = {label: False for label in object_labels}

            # Draw bounding boxes on the frame
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, confidence, class_idx = detection
                label = model.names[int(class_idx)]

                # Check if the label is in the object_labels list
                if label in object_labels:
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Add label and confidence score
                    text = f'{label}: {confidence:.2f}'
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Track detected objects
                    object_detected[label] = True

            # Update detected objects and their durations
            for label in object_labels:
                if object_detected[label]:
                    if detected_object != label:
                        detected_object = label
                        start_time = frame_count  # Use frame count instead of timestamp for simplicity
                else:
                    if detected_object == label:
                        end_time = frame_count
                        duration = end_time - start_time
                        total_duration[label] += duration
                        print(f'{label} detected for {duration} frames')
                        detected_object = None

            # Append processed frame to list
            processed_frames.append(frame)

        # Convert frame duration to seconds
        total_duration_seconds = {label: duration / fps for label, duration in total_duration.items()}

        # Print total duration of the detected objects in seconds
        for label, duration in total_duration_seconds.items():
            print(f'Total duration of {label}: {duration:.2f} seconds')

        # Save processed frames to folder
        processed_frames_dir = '/content/processed_framesfixbgtplis'
        if not os.path.exists(processed_frames_dir):
            os.makedirs(processed_frames_dir)

        for i, frame in enumerate(processed_frames):
            cv2.imwrite(f'{processed_frames_dir}/frame_{i:04d}.jpg', frame)

        # Release the OpenCV window
        cv2.destroyAllWindows()

        return processed_frames, total_duration_seconds
    
    # Call process_frames
    detection_results = mediapipe_detection(folder_path)
    
    # Call detect_objects_from_folder
    processed_frames, total_duration_seconds = detect_objects_from_folder(model_path, folder_path, object_labels, fps)
    
    return detection_results, processed_frames, total_duration_seconds

# Example Usage
model_path = '/content/weightsbest (1).pt'  # Path to model
folder_path = '/content/frames'  # Path to folder containing frames
object_labels = ['bottle', 'cigarette', 'phone', 'smoke', 'vape']  # Object labels to detect
fps = 30  # Frame rate of the video

detection_results, processed_frames, durations = combined_detection(model_path, folder_path, object_labels, fps)

# Print or further process the combined results
print('Detection Results:', detection_results)
print('Total Duration of Objects:', durations)

print('Durations for each object:')
for label, duration in durations.items():
    print(f'{label}: {duration:.2f} seconds')

