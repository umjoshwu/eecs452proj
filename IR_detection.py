import cv2
import numpy as np

# 初始化摄像头（USB摄像头通常为0）
cap = cv2.VideoCapture(0)

# 设置分辨率（支持的格式取决于摄像头）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
            
        # 显示实时画面
        cv2.imshow('USB Camera', frame)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
