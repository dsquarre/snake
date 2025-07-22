import cv2
from deepface import DeepFace
import numpy as np

class Emotion:
    def reward(self,frames = 1):
        cam = cv2.VideoCapture(0)

        #width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        #height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #out = cv2.VideoWriter('camera.mp4', fourcc, 30.0, (width, height))
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
       
        emotion_result = []

        for _ in range(frames):
            ret,frame = cam.read()
            if not ret:
                print("error")
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = frame[y:y+h, x:x+w]
                try:
                    # Analyze emotions using DeepFace
                    # enforce_detection=False allows DeepFace to attempt analysis even if
                    # it doesn't confidently detect a face within the provided ROI.
                    # This is useful if you've already done face detection with OpenCV.
                    analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                    # DeepFace returns a list of dictionaries if multiple faces are found
                    # (even if you pass a single face_roi, it still returns a list of one)
                    if analysis:
                        dominant_emotion = analysis[0]['dominant_emotion']
                        
                        # Map dominant emotion to +1, -1, or 0
                        if dominant_emotion in ['happy', 'surprise']:
                            emotion_result.append(2) # Positive
                        elif dominant_emotion in ['angry', 'fear', 'sad', 'disgust']:
                            emotion_result.append(-1) # Negative
                        else: # neutral
                            emotion_result.append(-1) # Neutral
                        
                        #print(f"Dominant Emotion: {dominant_emotion}, Result: {emotion_result}")
                        # You can also access percentages: analysis[0]['emotion']

                        # Optional: Draw on the frame
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Error during DeepFace analysis: {e}")
                    emotion_result.append(0)
                    # If no face is detected by DeepFace, or other error, it might raise an exception
                    # You might want to keep emotion_result as 0 in this case.
            else:
                print("No face detected in the frame.")
                emotion_result.append(0) # No face, so consider it neutral or handle as appropriate

            #out.write(frame)
            # Display the captured frame
            #cv2.imshow('Camera', frame)
            #print(emotion_result)
            # Press 'q' to exit the loop
        cam.release()
        #out.release()
        cv2.destroyAllWindows()
        print(emotion_result)

        return self.dominant(emotion_result)
    
    def dominant(self,results):
        positives,negatives,neutral = 0,0,0
        for emotion in results:
            if emotion < 0:
                negatives+=1
            elif emotion > 0:
                positives += 1
            else:
                neutral +=1
        if negatives>positives:
            if negatives>neutral or negatives > 1:
                return -1
        elif positives>negatives:
            if positives>neutral or positives > 1:
                return 1
        return 0
    
