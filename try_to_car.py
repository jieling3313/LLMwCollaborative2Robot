import paramiko
import time
import sys

# 建立 SSH 連接
client = paramiko.SSHClient()

# 載入系統主機金鑰
client.load_system_host_keys()

# 設定缺失主機金鑰策略，若找不到主機金鑰則自動接受
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

stdin = stdout = stderr = None  # 定義流變數，防止在某些情況下未初始化的問題

try:
    # 連接到 Jetson 小電腦
    client.connect('192.168.1.172', username='jetson', password='yahboom', look_for_keys=False, allow_agent=False)
    print("成功連接到 Jetson 小電腦")

    # 執行命令，並啟用 X11 轉發
    stdin, stdout, stderr = client.exec_command('/bin/python /home/jetson/Desktop/car_code/try_run.py', get_pty=True)

    # 持續監聽輸入並處理
    while True:
        # # 實時等待用戶輸入
        # user_input = input("請輸入方向指令 (F: 前進, B: 後退, L: 左轉, R: 右轉, LF: 左前, RF: 右前, LB: 左後, RB: 右後, LC: 左旋, RC: 右旋, S: 停止, exit: 退出): ")
        command = sys.stdin.readline().strip()  # 讀取主程式的指令

        if command.lower() == "exit":
            print("退出程式")
            stdin.write("exit\n")  # 輸入退出指令
            stdin.flush()  # 確保輸入已經發送
            break
        
        if command:  # 確保輸入非空
            stdin.write(f"{command}\n")  # 寫入指令
            stdin.flush()  # 確保輸入已經發送

        # 監控是否有新的輸出數據
        while stdout.channel.recv_ready():
            output = stdout.channel.recv(1024).decode()  # 持續讀取輸出
            print(output, end="")  # 顯示輸出

        # 讀取錯誤輸出並顯示（如果有錯誤）
        if stderr.channel.recv_ready():
            error_output = stderr.read().decode()
            if error_output:
                print("錯誤發生：")
                print(error_output)
    
except paramiko.AuthenticationException:
    print("SSH 驗證失敗，請檢查用戶名和密碼")
except paramiko.SSHException as e:
    print(f"SSH 連接或執行錯誤: {e}")
except Exception as e:
    print(f"發生了未知錯誤: {e}")
finally:
    # 確保流對象已初始化且關閉
    if stdin:
        stdin.close()
    if stdout:
        stdout.close()
    if stderr:
        stderr.close()

    # 關閉連接
    client.close()
    print("SSH 連接已關閉")
