import mediapipe as mp
import cv2 as cv
import numpy as np

# This will detect all points present
mp_pose = mp.solutions.pose
mp_marks = mp.solutions.drawing_utils  # marking all points


# taking video file input and setting parameters
cap = cv.VideoCapture('KneeBendVideo.mp4')
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
size = (width, height)
fps = int(cap.get(cv.CAP_PROP_FPS))

# counter and state parameters
free_counter, counter, use_counter = 0, 0, 0  # to store the counts of movements
advise, stage = None, None
images = []


def angle_finder(p1, p2, p3):
    # when finding angle there must be 3 points like   /\
    #                                                 /  \
    p1 = np.array(p1)  # Start
    p2 = np.array(p2)  # Middle
    p3 = np.array(p3)  # End

    radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) -\
        np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    degree = np.abs(radians*180.0/np.pi)

    if degree > 180.0:
        degree = 360 - degree  # if greater than reflex than its complimet
    return degree


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        # for mediapipe to detect it it is nedded to be in RGB fromat which is default BGR
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            #  coordinates of needed  landmarks (23, 25, and 27)
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculating Angle
            angle = angle_finder(hip, knee, ankle)

            # Render Detections
            a0 = int(ankle[0] * width)
            a1 = int(ankle[1] * height)

            k0 = int(knee[0] * width)
            k1 = int(knee[1] * height)

            h0 = int(hip[0] * width)
            h1 = int(hip[1] * height)

            cv.line(image, (h0, h1), (k0, k1), (255, 255, 0), 2)
            cv.line(image, (k0, k1), (a0, a1), (255, 255, 0), 2)
            cv.circle(image, (h0, h1), 5, (0, 0, 0), cv.FILLED)
            cv.circle(image, (k0, k1), 5, (0, 0, 0), cv.FILLED)
            cv.circle(image, (a0, a1), 5, (0, 0, 0), cv.FILLED)
            cv.putText(image, str(round(angle, 4)), tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)

            use_time = (1 / fps) * use_counter
            free_time = (1 / fps) * free_counter

            # Counter Logic
            if angle > 140:
                free_counter += 1
                use_counter = 0  # if greater than 140 then not counted
                stage = "Relaxed"
                advise = ""

            if angle < 140:
                free_counter = 0
                use_counter += 1  # if less than 140 then counted
                stage = "Bent"
                advise = ""

            # if time exceeds 8 sec then incremting count
            if use_time == 8:
                counter += 1
                advise = 'Rep completed'

            elif use_time < 8 and stage == 'Bent':  # if in position but 8 seconds have not passed
                advise = 'Keep Your Knee Bent'

            else:
                advise = " "  # if at rest it will display nothing

        except:
            pass

        # Setup status box
        cv.rectangle(image, (0, 0), (int(width), 60), (255, 255, 0), -1)

        # Rep data
        cv.putText(image, 'REPS', (10, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.putText(image, str(counter),
                   (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # Stage data
        cv.putText(image, 'STAGE', (105, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(image, stage,
                   (105, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # advise
        cv.putText(image, 'advise', (315, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.putText(image, advise,
                   (315, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # Bent Time
        cv.putText(image, 'BENT TIME', (725, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.putText(image, str(round(use_time, 2)),
                   (725, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # adding every image to array which will then i will add to video
        images.append(image)

        cv.imshow('Knee Bend Excercise', image)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# Generating the video with commands
res = cv.VideoWriter('Result.mp4', cv.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(images)):
    res.write(images[i])
res.release()
