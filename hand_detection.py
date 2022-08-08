import cv2
import mediapipe as mp
import time
import numpy as np

# 定义规则函数
def get_str_guester(up_fingers, list_lms):
    if len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:  # 如果只有个点在凸包外，并且这个点的编号是4
        str_guester = "scissors"
    elif len(up_fingers) == 5 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 12 and up_fingers[3] == 16 and up_fingers[4] == 20:
        str_guester = "paper"
    # elif len(up_fingers) == 0:
    #     str_guester = "rock"
    elif len(up_fingers) == 0:
        str_guester = "rock"
    else:
        str_guester = ""

    return str_guester
    # if len(up_fingers)==1 and up_fingers[0]==8:
        # v1 = list_lms[6]-list_lms[7]    # 算矢量

pTime = 0   # previoustime之前的时间
cTime = 0   # currenttime当前的时间
# 新建窗口
image_width = 640
image_height = 480
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('video', image_width, image_height)

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands    # 使用手部的模型
hands = mpHands.Hands()     # 参数(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils # 画手的坐标
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)  # 点的设置
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=4)   # 线的设置


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 读到图像
    # bgr 转 rgb给mediapipe用
    frame = cv2.flip(frame, 1)  # 镜像操作
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #转换为RGB
    result = hands.process(imgRGB)
    # print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)
        # 把所有侦测到的手画出来
        # for handLms in result.multi_hand_landmarks:
        #     mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
        #     # 得到手的各点坐标
        #     for i, lm in enumerate(handLms.landmark):
        #         xPos = int(lm.x * image_width)       # lm.x,lm.y是对应点x,y坐标占窗口的比例，乘回窗口长宽得到点的坐标
        #         yPos = int(lm.y * image_height)
        #         # 在图上标出点的下标
        #         if i == 12:
        #             cv2.circle(frame, (xPos, yPos), 20, (5, 96, 36), cv2.FILLED)
        #         #  在图像上显示各点的下标
        #         cv2.putText(frame, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        #         print(i, xPos, yPos)
        # 采集所有关键点的坐标
        list_lms = []
        for i in range(21):
            pos_x = hand.landmark[i].x*image_width
            pos_y = hand.landmark[i].y*image_height
            list_lms.append([int(pos_x), int(pos_y)])   # 将点的坐标加入list_lms中

        # 构造凸包点
        list_lms = np.array(list_lms, dtype=np.int32)
        hull_index = [0, 1, 2, 3, 6,10, 14, 19, 18, 17, 10]
        hull = cv2.convexHull(list_lms[hull_index, :])
        # 绘制凸包
        cv2.polylines(frame, [hull], True, (0, 255, 0), 2)

        # 查找外部的点数
        n_fig = -1
        ll = [4, 8, 12, 16, 20]     # 凸包外的关键点
        up_fingers = []  # 检测结果列表

        for i in ll:
            pt = (int(list_lms[i][0]), int(list_lms[i][1]))   # 求出点的坐标
            dist = cv2.pointPolygonTest(hull, pt, True)     # 检测该点与凸包的位置关系
            if dist < 0:    # 点在凸包外面
                up_fingers.append(i)    # 添加这个点的坐标，在列表末尾添加新的对象
                # 获取手势的名称
        str_guester = get_str_guester(up_fingers, list_lms)     # 自定义函数参数（检测到的在凸包外的点, 关键点的坐标）
        cv2.putText(frame, '    %s'%(str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX,2, (255, 255, 0), 3, cv2.LINE_AA) # 打印识别出的结果



    # 计算FPS
    cTime = time.time()     # cTime=现在的时间
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS    :   {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# 释放摄像头
cap.release()
# 释放所有窗口
cv2.destroyAllWindows()