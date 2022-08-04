import cv2
import mediapipe as mp

# 新建窗口
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('video', 640, 480)

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands    # 使用手部的模型
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils # 画手的坐标
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)  # 点的设置
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=4)   # 线的设置


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 读到图像
    # bgr 转 rgb给mediapipe用
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    # print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        # 把所有侦测到的手画出来
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
            # 得到手的各点坐标
            for i, lm in enumerate(handLms.landmark):
                xPos = int(lm.x * 640)       # lm.x,lm.y是对应点x,y坐标占窗口的比例，乘回窗口长宽得到点的坐标
                yPos = int(lm.y * 480)
                # 在图上标出点的下标
                cv2.putText(frame, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                print(i, xPos, yPos)
    #frame = cv2.flip(frame, 1)  # 镜像操作
    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# 释放摄像头
cap.release()
# 释放所有窗口
cv2.destroyAllWindows()