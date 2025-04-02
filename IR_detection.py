import cv2
import numpy as np
import time

# 树莓派专用配置
PI_CAMERA_INDEX = 0        # CSI摄像头用0，USB摄像头通常为1
RESOLUTION = (640, 480)    # 根据性能可降为(320,240)
FRAME_RATE = 15            # 目标帧率
IR_THRESHOLD = 180         # 红外反射阈值
MIN_AREA = 20              # 最小检测区域（像素）
SHOW_DEBUG = True           # 显示处理过程

def pi_camera_init():
    """树莓派摄像头初始化优化"""
    cap = cv2.VideoCapture(PI_CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头，请检查：\n1.摄像头连接\n2.用户权限(video组)\n3.设备索引号")

    # 设置MJPG格式提升性能
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    return cap

def detect_hidden_camera(frame):
    """改进型隐藏摄像头检测算法"""
    # 降采样提升处理速度
    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    
    # 转换为HSV空间过滤高亮区域
    hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0,0,200), (180,30,255))
    
    # 红外特征增强
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # 动态阈值处理
    _, thresh = cv2.threshold(gray, IR_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # 优化形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 轮廓分析
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            # 坐标恢复（补偿降采样）
            x,y,w,h = [v*2 for v in cv2.boundingRect(cnt)]
            
            # 几何特征验证
            aspect_ratio = float(w)/h
            if 0.7 < aspect_ratio < 1.3:
                detected = True
                if SHOW_DEBUG:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
    
    return detected, frame

def main():
    cap = pi_camera_init()
    last_alarm = 0  # 报警状态保持
    
    try:
        while True:
            start_time = time.time()
            
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("视频流中断")
                break
                
            # 执行检测
            detected, processed_frame = detect_hidden_camera(frame)
            
            # 报警逻辑
            if detected:
                last_alarm = time.time()
                cv2.putText(processed_frame, "CAMERA DETECTED!", (20,40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            elif time.time() - last_alarm < 2:  # 报警保持2秒
                cv2.putText(processed_frame, "ALARM ACTIVE", (20,40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            
            # 性能监控
            fps = 1/(time.time() - start_time + 1e-6)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (20,70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
            # 显示结果
            if SHOW_DEBUG:
                cv2.imshow("Hidden Camera Detection", processed_frame)
            
            # 退出控制
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
