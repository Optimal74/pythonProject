import torch
from ultralytics import YOLO
import supervision as sv
import cv2
import telebot
import datetime
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def main():
    token = '6641634996:AAG8ZCJRsv9CF8WK2yt886T4inC-OXqjoe4'
    bot = telebot.TeleBot(token)


    model = YOLO("yolov8m-pose.pt")
    model.fuse()
    box_annotator = sv.BoxAnnotator(thickness=2,
                                    text_thickness=1,
                                    text_scale=0.5)
    START = sv.Point(320, 0)
    END = sv.Point(320, 480)

    line_zone = sv.LineZone(start=START, end=END)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2,
                                               text_thickness=1,
                                               text_scale=0.5)
    last_in_count = None

    for result in model.track(source='rtsp://admin:admin@192.168.100.168:554/1',
                              stream=True,
                              tracker='bytetrack.yaml'):

        frame = result.orig_img
        detections = sv.Detections.from_ultralytics(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            keypoints = result.keypoints.data.cpu().numpy()[0].astype(int)
            for i, point in enumerate(keypoints):
                if i == 9 or i == 10:
                    if point[1] < keypoints[11][1] and point[1] < keypoints[12][1]:
                        is_swinging = True
                        print(f'{point[1]}, {keypoints[11][1]} and {point[1]}, {keypoints[12][1]}')
                cv2.circle(frame, tuple(point[:2]), 2, (0, 0, 255), -1)
                cv2.putText(frame, str(i), tuple(point[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        detections = detections[detections.class_id == 0]

        labels = [
            f'#{tracker_id } {confidence:0.2f} {model.model.names[class_id]}' for _, _, confidence, class_id, tracker_id
            in detections
        ]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        line_zone.trigger(detections=detections)
        line_zone_annotator.annotate(frame=frame, line_counter=line_zone)
        #print("In:", line_zone.in_count)
        #print("Out:", line_zone.out_count)

        if last_in_count is None or last_in_count != line_zone.in_count:
            current_time = datetime.datetime.now().strftime("%H:%M")
            if line_zone.in_count >= 1:
                #message = f'Прыжок: № {line_zone.in_count}. Время: {current_time}'
                #bot.send_message(chat_id='2102106376', text=message)
                last_in_count = line_zone.in_count


        cv2.imshow('yolo8s', frame)
        if (cv2.waitKey(30) == 27):
            break
if __name__ == "__main__":
    main()