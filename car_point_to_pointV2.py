###########################################
##      這邊我修改了一下他的微調方式      ##
##      可以搜尋 微調 應該就找的到了ㄖ    ##
###########################################

import numpy as np
import cv2
import cv2.aruco as aruco
import time
import os
import math
import subprocess

# 定義攝影機參數（內參矩陣和畸變參數）
mtx = np.array([
    [1.12081283e+03, 0.00000000e+00, 6.24450024e+02],
    [0.00000000e+00, 1.12054499e+03, 3.97267878e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

dist = np.array([1.32226416e-01, -1.07953073e+00, -3.87129791e-03, 3.61411900e-04, 2.06248060e+00])  # 假設畸變係數為零（可以替換為校正後的參數）

# 開啟攝影機（默認攝影機，索引為 0）
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("無法打開攝影機")
    exit()

# ArUco 字典與檢測參數
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# 字體設定（用於顯示 ID 或錯誤信息）
font = cv2.FONT_HERSHEY_SIMPLEX

# 先設原點為 (0,0,0)
tvec_0 = None

# 儲存 ID 4 和 ID 5 的位置和方向
car_4_tvec = None
car_5_tvec = None
car_4_rvec = None
car_5_rvec = None

# # 設定目標座標
# target_x, target_y = -0.6, -0.6  # 目標位置（以米為單位）
target_x, target_y = None, None

# 啟動機器人控制程式
process = subprocess.Popen(['python3', 'try_version_and_move/try_to_car.py'],
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           text=True)

old_cmd = None

if_fine_tune_4 = False
if_fine_tune_5 = False

def command_to_car(cmd):
    global old_cmd
    if old_cmd != cmd:
        old_cmd = cmd
        print(f"[主程式] 發送指令: {cmd}")
        process.stdin.write(cmd + "\n")  # 傳送指令
        process.stdin.flush()  # 確保指令立即送出

def calculate_direction_and_move(car_id, car_tvec, rvec_car, target_x, target_y, other_car_tvec=None):
    """
    計算車子相對於目標的位置和朝向，並返回所需的移動指令。
    
    :param car_id: 車子的 ID（4 或 5）。
    :param car_tvec: 該車子的相對座標（np.array([x, y, z])）。
    :param rvec_car: 車子的旋轉向量 (rvec)。
    :param target_x: 目標的 X 座標（相對於原點）。
    :param target_y: 目標的 Y 座標（相對於原點）。
    :param other_car_tvec: 另一台車的座標（如果有），避免碰撞。
    :return: 方向指令（"F", "B", "L", "R", "LF", "RF", "LB", "RB", "S"）。
    """
    global if_fine_tune_4
    global if_fine_tune_5

    if car_tvec is None or target_x is None or target_y is None:
        command_to_car("S")
        return "S"  # 若車子座標未知，則停止

    car_x, car_y, _ = car_tvec  # 取得車子的 (x, y) 位置

    # 計算車子與目標的距離
    dx = target_x - car_x
    dy = target_y - car_y
    distance_to_target = math.sqrt(dx**2 + dy**2)

    # 設定距離閾值，若車子靠近目標一定範圍則停止
    stop_threshold = 0.3  # 30cm 內視為到達目標

    # 設定兩車最小距離
    min_distance_between_cars = 0.35  # 40cm 內視為太接近

    # 如果車子已到目標範圍內，則停止
    if distance_to_target < stop_threshold:
        command_to_car("S")
        if car_id == 4:
            if_fine_tune_4 = True
        if car_id == 5:
            if_fine_tune_5 = True
        return "S"
    else:
        if car_id == 4:
            if_fine_tune_4 = False
        if car_id == 5:
            if_fine_tune_5 = False

    # 計算車子與目標點的夾角 (目標方向)
    target_angle = math.degrees(math.atan2(dy, dx))

    # 取得車子的朝向角度
    rvec_matrix, _ = cv2.Rodrigues(rvec_car)
    forward_vector = rvec_matrix[:, 0]  # 改用 x 軸方向（車頭方向）
    car_angle = math.degrees(math.atan2(forward_vector[1], forward_vector[0]))

    # 計算車頭朝向與目標方向的角度差
    angle_diff = (target_angle - car_angle) % 360

    # 若有另一台車，避免碰撞
    if other_car_tvec is not None:
        other_x, other_y, _ = other_car_tvec
        distance_to_other_car = math.sqrt((car_x - other_x) ** 2 + (car_y - other_y) ** 2)
        
        # 如果距離過近，則優先避免碰撞
        if distance_to_other_car < min_distance_between_cars:
            command_to_car("S")
            return "S"

    # **根據角度選擇方向**
    if 0 <= angle_diff < 22.5 or 337.5 <= angle_diff <= 360:
        command_to_car("F")
        return "F"  # 正前
    elif 22.5 <= angle_diff < 67.5:
        command_to_car("RF")
        return "RF"  # 右前
    elif 67.5 <= angle_diff < 112.5:
        command_to_car("R")
        return "R"  # 右
    elif 112.5 <= angle_diff < 157.5:
        command_to_car("RB")
        return "RB"  # 右後
    elif 157.5 <= angle_diff < 202.5:
        command_to_car("B")
        return "B"  # 正後
    elif 202.5 <= angle_diff < 247.5:
        command_to_car("LB")
        return "LB"  # 左後
    elif 247.5 <= angle_diff < 292.5:
        command_to_car("L")
        return "L"  # 左
    elif 292.5 <= angle_diff < 337.5:
        command_to_car("LF")
        return "LF"  # 左前

    command_to_car("S")
    return "S"  # 預設停止


##微調
def fine_tune_position_and_orientation(car_id, car_tvec, rvec_car, target_x, target_y):
    """
    微調車子：
    - 當機器人距離目標小於 30cm 時，嘗試將機器人導引到以目標為圓心、半徑 15cm 的外切點，
      並使機器人右側方向(+Y軸)正對目標。
    - 若機器人距離目標 < 15cm，就視為已經到達或太近，直接停止。
    - (示範性質) 實際應該用更細膩的動作控制，不斷更新位置再發指令。
    """
    if car_tvec is None or target_x is None or target_y is None:
        return "S"  # 無法微調

    # 取得機器人目前座標
    car_x, car_y, _ = car_tvec

    # 計算目標距離 d
    dx = target_x - car_x
    dy = target_y - car_y
    d = math.sqrt(dx**2 + dy**2)

    # 若大於 30cm，尚未進入「微調」範圍
    if d >= 0.30:
        return "fine_tune_position"

    # 若小於等於 15cm，視為「已經抵達、或太近」，直接停止
    if d <= 0.15:
        command_to_car("S")
        return "S"

    # -------------------------
    # 以下為「外切點 + 右側對準」之幾何計算
    # -------------------------
    # 1) 計算當前車頭角度 car_angle
    rvec_matrix, _ = cv2.Rodrigues(rvec_car)
    forward_vector = rvec_matrix[:, 0]  # x 軸方向
    car_angle = math.degrees(math.atan2(forward_vector[1], forward_vector[0]))

    # 2) 計算「以目標為圓心、半徑 = 0.15」的兩個外切點 (p1, p2)
    #    幾何：
    #    d = |R - T| (機器人到目標的距離)
    #    alpha = arccos(r / d), 其中 r=0.15
    #    baseAngle = angle(T->R) = atan2(Ry - Ty, Rx - Tx)
    #    p1, p2 的方向角 = baseAngle ± alpha
    #    p1 = T + r * [cos(baseAngle + alpha), sin(baseAngle + alpha)]
    #    p2 = T + r * [cos(baseAngle - alpha), sin(baseAngle - alpha)]
    radius = 0.15
    tx, ty = target_x, target_y
    rx, ry = car_x, car_y

    baseAngle = math.atan2(ry - ty, rx - tx)  # T->R
    alpha = math.acos(radius / d)             # 外切角

    theta1 = baseAngle + alpha
    p1x = tx + radius * math.cos(theta1)
    p1y = ty + radius * math.sin(theta1)

    theta2 = baseAngle - alpha
    p2x = tx + radius * math.cos(theta2)
    p2y = ty + radius * math.sin(theta2)

    # 3) 計算各切點對應的「車頭理想角度」(車右側正對目標 => 車頭角度 = angle(p->t) - 90°)
    def desired_heading_for_point(px, py, tx, ty):
        vx = tx - px
        vy = ty - py
        angle_pt = math.degrees(math.atan2(vy, vx))  # p->t
        return (angle_pt - 90) % 360

    heading_p1 = desired_heading_for_point(p1x, p1y, tx, ty)
    heading_p2 = desired_heading_for_point(p2x, p2y, tx, ty)

    # 計算角度差(回傳範圍 -180~180)
    def angle_diff(a, b):
        diff = (a - b + 180) % 360 - 180
        return diff

    diff1 = angle_diff(heading_p1, car_angle)
    diff2 = angle_diff(heading_p2, car_angle)

    # 4) 選擇旋轉量較小者
    abs1 = abs(diff1)
    abs2 = abs(diff2)
    if abs1 < abs2:
        chosen_px, chosen_py = p1x, p1y
        chosen_heading = heading_p1
        chosen_diff = diff1
    else:
        chosen_px, chosen_py = p2x, p2y
        chosen_heading = heading_p2
        chosen_diff = diff2

    # 5) 決定動作 (只示範一次性)
    rotate_threshold = 5.0   # 大於此角度差 => 先旋轉
    move_threshold = 0.02    # 與切點距離小於 2cm 視為「已到達切點」

    dx_c = chosen_px - rx
    dy_c = chosen_py - ry
    dist_c = math.sqrt(dx_c**2 + dy_c**2)

    # 若角度差大，先旋轉
    if abs(chosen_diff) > rotate_threshold:
        if chosen_diff > 0:
            command_to_car("RC")  # 右旋
            return "微調旋轉(右)"
        else:
            command_to_car("LC")  # 左旋
            return "微調旋轉(左)"
    else:
        # 角度已大致正確 => 檢查與切點距離
        if dist_c > move_threshold:
            # 判斷是前進還是後退
            path_angle = math.degrees(math.atan2(dy_c, dx_c))   # R->p
            drive_diff = angle_diff(path_angle, car_angle)      # path_angle - car_angle
            # 若夾角 <= 90 => 前進，否則後退
            if abs(drive_diff) <= 90:
                command_to_car("F")
                return "微調前進"
            else:
                command_to_car("B")
                return "微調後退"
        else:
            # 已經接近切點 => 做最後細微旋轉 or 停止
            final_diff = angle_diff(chosen_heading, car_angle)
            if abs(final_diff) > rotate_threshold / 2:
                if final_diff > 0:
                    command_to_car("RC")
                else:
                    command_to_car("LC")
                return "最後微調旋轉"
            # 完成
            command_to_car("S")
            return "S"

    # 預設
    return "fine_tune_position"

def draw_target_position(frame, target_x, target_y, tvec_0, mtx, dist):
    """
    在畫面上繪製目標位置，目標位置是相對於 ID 0（原點）的座標。
    :param frame: 影像幀
    :param target_x: 目標相對於 ID 0 的 x 座標（單位：m）
    :param target_y: 目標相對於 ID 0 的 y 座標（單位：m）
    :param tvec_0: ID 0（原點）的平移向量
    :param mtx: 相機內參矩陣
    :param dist: 相機畸變參數
    """
    if tvec_0 is None or target_x is None or target_y is None:
        return  # 如果沒有偵測到原點，則無法繪製目標位置
    
    # 計算目標的世界座標 = 原點座標 + 目標相對座標
    target_world_x = tvec_0[0][0] + target_x
    target_world_y = tvec_0[0][1] + target_y
    target_world_z = tvec_0[0][2]  # 假設與 ID 0 同高度（Z 軸不變）

    # 目標點在世界座標系中的 3D 坐標
    target_3d_point = np.array([[target_world_x, target_world_y, target_world_z]], dtype=np.float32)

    # 轉換為 2D 影像座標
    image_points, _ = cv2.projectPoints(target_3d_point, np.zeros((3, 1)), np.zeros((3, 1)), mtx, dist)

    # 獲取 2D 影像座標
    target_px, target_py = int(image_points[0][0][0]), int(image_points[0][0][1])

    # 繪製紅色圓點表示目標位置
    cv2.circle(frame, (target_px, target_py), 10, (0, 0, 255), -1)
    cv2.putText(frame, "Target", (target_px + 10, target_py - 10), font, 0.6, (0, 0, 255), 2)

print("按下 'q' 鍵退出程式")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("攝影機讀取失敗")
            break

        # 影像尺寸
        h, w = frame.shape[:2]

        # 校正相機影像
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # 灰階處理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 檢測 ArUco 標記
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # 進行姿態估算
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.055, mtx, dist)

            for i in range(len(ids)):
                # 繪製檢測到的標記
                aruco.drawDetectedMarkers(frame, corners)

                # 繪製每個標記的坐標軸
                cv2.drawFrameAxes(frame, mtx, dist, rvecs[i], tvecs[i], 0.1)

                # 顯示 ID
                text = f"ID: {ids[i][0]}"
                position = (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10)
                cv2.putText(frame, text, position, font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                # 顯示座標
                # 如果是 ID 0，將其視為原點
                if ids[i][0] == 0:
                    tvec_0 = tvecs[i]  # 儲存 ID 0 的平移向量
                    cv2.putText(frame, "Origin (ID 0)", (50, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if tvec_0 is not None:
                    # 計算相對於 ID 0 的座標
                    relative_tvec = tvecs[i] - tvec_0
                    x, y, z = relative_tvec[0][0], relative_tvec[0][1], relative_tvec[0][2]
                    coordinate_text = f"Coord: ({x:.2f}, {y:.2f}, {z:.2f})"
                    position = (int(corners[i][0][0][0]), int(corners[i][0][0][1]) + 20)
                    cv2.putText(frame, coordinate_text, position, font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    # 給車子座標
                    if ids[i][0] == 4:
                        car_4_tvec = (x, y, z)
                        car_4_rvec = rvecs[i]
                        
                        # 計算旋轉矩陣
                        rvec_matrix_4, _ = cv2.Rodrigues(car_4_rvec)
                        
                        # 取得車頭方向向量 (x 軸方向)
                        forward_vector_4 = rvec_matrix_4[:, 0]  # 第一列代表 x 軸方向
                        
                        # 沿著車頭方向向前補償 6 公分 (0.06m)
                        car_4_tvec = (
                            car_4_tvec[0] + forward_vector_4[0] * 0.06,
                            car_4_tvec[1] + forward_vector_4[1] * 0.06,
                            car_4_tvec[2] + forward_vector_4[2] * 0.06
                        )

                    if ids[i][0] == 5:
                        car_5_tvec = (x, y, z)
                        car_5_rvec = rvecs[i]
                        
                        # 計算旋轉矩陣
                        rvec_matrix_5, _ = cv2.Rodrigues(car_5_rvec)
                        
                        # 取得車頭方向向量 (x 軸方向)
                        forward_vector_5 = rvec_matrix_5[:, 0]  # 第一列代表 x 軸方向
                        
                        # 沿著車頭方向向前補償 6 公分 (0.06m)
                        car_5_tvec = (
                            car_5_tvec[0] + forward_vector_5[0] * 0.06,
                            car_5_tvec[1] + forward_vector_5[1] * 0.06,
                            car_5_tvec[2] + forward_vector_5[2] * 0.06
                        )
                    
                    if ids[i][0] == 6:
                        target_x, target_y = x, y
                        
                # 計算移動方向
                if car_4_tvec is not None:
                    if if_fine_tune_4 == False:
                        move_command_4 = calculate_direction_and_move(4, car_4_tvec, car_4_rvec, target_x, target_y, car_5_tvec)
                        cv2.putText(frame, f"Move 4: {move_command_4}", (50, 200), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    else:
                        move_command_4 = fine_tune_position_and_orientation(4, car_4_tvec, car_4_rvec, target_x, target_y)
                        cv2.putText(frame, f"Move 4: {move_command_4}", (50, 200), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

                if car_5_tvec is not None:
                    if if_fine_tune_5 == False:  
                        move_command_5 = calculate_direction_and_move(5, car_5_tvec, car_5_rvec, target_x, target_y, car_4_tvec)
                        cv2.putText(frame, f"Move 5: {move_command_5}", (50, 250), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    else:
                        fine_tune_position_and_orientation(5, car_5_tvec, car_5_rvec, target_x, target_y)
                        cv2.putText(frame, f"Move 5: {move_command_5}", (50, 250), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

        else:
            # 如果沒有檢測到標記，顯示 "No IDs"
            cv2.putText(frame, "No IDs", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        draw_target_position(frame, target_x, target_y, tvec_0, mtx, dist)

        # 顯示影像
        cv2.imshow("ArUco Detection", frame)

        # 按下 'q' 鍵退出程式
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("退出程式")
            break

        # 按下空格鍵保存影像
        if key == ord(' '):
            if not os.path.exists('calibration_images'):
                os.makedirs('calibration_images')

            filename = os.path.join('calibration_images', f"{int(time.time())}.jpg")
            cv2.imwrite(filename, frame)
            print(f"保存影像到 {filename}")

        pass
except Exception as e:
    print(f"發生錯誤: {e}")
finally:
    # 釋放攝影機並關閉視窗
    cap.release()
    cv2.destroyAllWindows()
    if process.poll() is None:  
        process.stdin.write("EXIT\n")   # 傳送指令
        process.stdin.flush()           # 確保指令立即送出
        # 確保子程式完全結束
        process.stdin.close()           # 關閉標準輸入，告知子程序沒有更多輸入
        process.wait()
    print("資源已釋放，程式結束")

