import cv2
import numpy as np
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

# 1. إعداد النموذج مع خفض الثقة إلى 30% (0.3) ليكون فائق الحساسية
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(0)
print("جاري تشغيل الكاميرا... اضغط 'q' للخروج.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # عكس الصورة كأنها مرآة
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # تحويل الألوان ليقرأها النموذج
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    box_points = []
    hands_count = 0

    if results.multi_hand_landmarks:
        hands_count = len(results.multi_hand_landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            # رسم هيكل اليد كنقاط حمراء/خضراء لتتأكد أن الكاميرا تراك
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # استخراج أطراف الإبهام والسبابة
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            box_points.append((int(index_tip.x * w), int(index_tip.y * h)))
            box_points.append((int(thumb_tip.x * w), int(thumb_tip.y * h)))

    # طباعة عدد الأيدي المكتشفة على الشاشة للمساعدة
    cv2.putText(frame, f"Hands detected: {hands_count}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 2. تطبيق التأثير بذكاء إذا تم كشف 4 نقاط (يدين معاً)
    if len(box_points) >= 4:
        # تحويل النقاط إلى مصفوفة
        pts = np.array(box_points, dtype=np.int32)
        
        # استخدام Convex Hull لترتيب النقاط ومنع تداخل الإطار أو تشوهه
        hull = cv2.convexHull(pts)
        
        # إنشاء القناع (Mask)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        # إنشاء التأثير (النيجاتيف)
        inverted_frame = cv2.bitwise_not(frame)
        
        # دمج التأثير داخل الإطار فقط
        frame = np.where(mask[:, :, np.newaxis] == 255, inverted_frame, frame)
        
        # رسم خطوط الإطار باللون الأخضر السماوي (Cyan) ليكون واضحاً
        cv2.polylines(frame, [hull], isClosed=True, color=(255, 255, 0), thickness=4)

    # عرض النتيجة النهائية
    cv2.imshow('Smart Hand Frame Filter', frame)

    # الخروج عند الضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()