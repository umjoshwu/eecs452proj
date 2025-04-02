import cv2
import numpy as np

# 摄像头参数设置
THRESHOLD = 200       # 亮度阈值（0-255，根据实际情况调整）
MIN_AREA = 10         # 最小区域像素数（过滤小光点）
SHOW_DEBUG = True     # 显示调试窗口
ALARM_TEXT = "Hidden Camera Detected!"

def detect_ir_signals():
    # 初始化摄像头
    cap = cv2.VideoCapture(1)  # 0表示默认摄像头，如有多个摄像头需调整编号
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # 设置摄像头参数（根据具体硬件调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 创建背景减法器（用于动态环境）
    fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 动态阈值处理
        _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # 形态学操作（去除噪点）
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 分析轮廓
        detected = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                x,y,w,h = cv2.boundingRect(cnt)
                
                # 特征筛选（可根据实际情况添加更多条件）
                aspect_ratio = float(w)/h
                if 0.8 < aspect_ratio < 1.2:  # 近似正方形的光点更可疑
                    detected = True
                    if SHOW_DEBUG:
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        
        # 显示报警
        if detected:
            cv2.putText(frame, ALARM_TEXT, (20,40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # 显示调试窗口
        if SHOW_DEBUG:
            cv2.imshow("IR Detection", frame)
            cv2.imshow("Threshold", thresh)
            cv2.imshow("Cleaned", cleaned)
            
        # 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_ir_signals()