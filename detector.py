import cv2
import mediapipe as mp 
import joblib
import numpy as np 


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.80, min_tracking_confidence=0.80)


def data_clean(landmark):
  data = landmark[0]
  try:
    data = str(data)
    data = data.strip().split('\n')
    garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']
    without_garbage = []
    for i in data:
        if i not in garbage:
            without_garbage.append(i)
    clean = []
    for i in without_garbage:
        i = i.strip()
        clean.append(i[2:])
    for i in range(0, len(clean)):
        clean[i] = float(clean[i])
    return([clean])
  except:
    return(np.zeros([1,63], dtype=int)[0])  



def detection(success, image):
  global letter
  while True:
    image = cv2.flip(image, 1)
    if not success:
      break
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      cleaned_landmark = data_clean(results.multi_hand_landmarks)
      if cleaned_landmark:
        clf = joblib.load('new_ASL_model.pkl')
        y_pred = clf.predict(cleaned_landmark)
        letter = str(y_pred[0])
        image = cv2.putText(image, str(y_pred[0]), (50,150), cv2.FONT_HERSHEY_SIMPLEX,  3, (0,0,255), 2, cv2.LINE_AA)     
    return image,letter


letter=''