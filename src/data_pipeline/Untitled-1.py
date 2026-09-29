import cv2
import mediapipe as mp
import numpy as np

# 1. تحميل نموذج التعلم العميق لتتبع اليد
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 2. تحويل الصورة وتمريرها للنموذج للاكتشاف
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # 3. استخراج إحداثيات أطراف الأصابع (رؤوس الإطار)
        points = []
        for hand_landmarks in results.multi_hand_landmarks:
            # استخراج نقطة رأس السبابة (Index Tip) ورأس الإبهام (Thumb Tip)
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            points.append((int(index_tip.x * w), int(index_tip.y * h)))
            
        # 4. رسم ماسك (Mask) بين نقاط اليد وتطبيق التأثير الداخلي (مثل العكس Invert)
        if len(points) >= 3:
            mask = np.zeros_like(frame)
            cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))
            
            # تطبيق الفلتر داخل المنطقة فقط
            inverted = cv2.bitwise_not(frame)
            frame = np.where(mask == 255, inverted, frame)

    cv2.imshow('Hand Frame Filter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()