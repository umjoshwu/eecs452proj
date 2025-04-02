import cv2
import numpy as np
import time

# 树莓派专用参数配置
PI_CAMERA_INDEX = 0     # 树莓派CSI摄像头用0，USB摄像头可能需要改为1
RESOLUTION = (640, 480) # 根据性能需求可降为(320,240)
FRAME_RATE = 15         # 帧率限制
THRESHOLD = 180         # 优化后的亮度阈值
MIN_AREA = 20           # 调整最小区域
SHOW_DEBUG = True       # 调试模式开关
ALARM_TEXT = "Hidden Camera Detected!"

def pi_camera_init():
    """树莓派专用摄像头初始化"""
    cap = cv2.VideoCapture(PI_CAMERA_INDEX)
    if not cap.isOpened():
        # 尝试V4L驱动模式
        cap = cv2.VideoCapture(PI_CAMERA_INDEX, cv2.CAP_V4L2)
    
    # 设置MJPG编码格式提升性能
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    return cap

def optimize_processing(frame):
    """树莓派专用处理优化"""
    # 降采样提升处理速度
    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    
    # 转换为灰度并应用CLAHE增强对比度
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 动态阈值处理
    _, thresh = cv2.threshold(enhanced, THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # 轻量级形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return cleaned, small_frame

def detect_ir_signals():
    # 初始化树莓派摄像头
    cap = pi_camera_init()
    if not cap.isOpened():
        print(f"无法打开摄像头，请检查：")
        print("1. 摄像头是否正确连接")
        print("2. 用户是否在video组（执行sudo usermod -aG video $USER）")
        print("3. 尝试修改PI_CAMERA_INDEX参数")
        return

    # 性能监控参数
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 读取帧并控制帧率
            ret, frame = cap.read()
            if not ret:
                print("视频流中断")
                break
            
            # 树莓派专用优化处理
            processed, small_frame = optimize_processing(frame)
            
            # 查找轮廓
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 分析轮廓
            detected = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_AREA:
                    # 恢复原始坐标
                    x,y,w,h = [v*2 for v in cv2.boundingRect(cnt)]  # 补偿降采样
                    
                    aspect_ratio = float(w)/h
                    if 0.7 < aspect_ratio < 1.3:  # 放宽宽高比范围
                        detected = True
                        if SHOW_DEBUG:
                            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            
            # 显示报警信息
            if detected:
                cv2.putText(frame, ALARM_TEXT, (20,40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            # 性能监控
            frame_count += 1
            if frame_count >= 10:
                fps = frame_count / (time.time() - start_time)
                print(f"当前FPS: {fps:.1f}")
                frame_count = 0
                start_time = time.time()
            
            # 显示输出
            if SHOW_DEBUG:
                cv2.imshow("IR Detection", frame)
                cv2.imshow("Processing", processed)
            
            # 退出控制
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_ir_signals()
