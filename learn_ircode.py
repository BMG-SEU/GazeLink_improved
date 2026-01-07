import broadlink
import time
import json
import os
from getpass import getpass
import socket

def get_local_ip():
    """获取本机 IP 地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def connect_device():
    local_ip = get_local_ip()
    print(f"本地 IP: {local_ip}")

    # 方法 1：自动发现
    try:
        print("尝试自动发现设备...")
        devices = broadlink.discover(
            timeout=50,  # 增加超时时间
            local_ip_address=local_ip,
            discover_ip_address="255.255.255.255"  # 使用通用广播地址
        )

        if devices:
            device = devices[0]
            device.auth()
            return device
        else:
            print("未发现设备")
    except Exception as e:
        print(f"自动发现失败: {e}")

    # 方法 2：手动连接
    try:
        print("尝试手动连接设备...")
        device = broadlink.hello(
            host=("172.20.10.2", 80),  # 替换为实际 IP
            mac=bytearray.fromhex("e8165606ed15")  # 替换为实际 MAC
        )
        device.auth()
        return device
    except Exception as e:
        print(f"手动连接失败: {e}")

    print("所有连接方式均失败")
    return None

# 检查并创建 ir_codes.json 文件
def ensure_ir_codes_file():
    if not os.path.exists("ir_codes.json"):
        with open("ir_codes.json", "w") as f:
            json.dump({}, f)
        print("已创建空的 ir_codes.json 文件")

# 学习新编码
def learn_ir_code(device, save_name):
    ensure_ir_codes_file()  # 确保文件存在
    print("进入学习模式...")
    device.enter_learning()
    time.sleep(1)  # 等待设备进入学习模式
    input(f"请对准RM4按下遥控器按钮，完成后按回车...")
    print("检查学习数据...")
    ir_packet = device.check_data()
    if ir_packet:
        print(f"成功接收到红外信号: {ir_packet.hex()}")
        with open("ir_codes.json", "r+") as f:
            codes = json.load(f)
            codes[save_name] = ir_packet.hex()
            f.seek(0)
            json.dump(codes, f, indent=4)
            f.truncate()
        print(f"已保存编码: {save_name}")
        return True
    else:
        print("未接收到红外信号")
        return False

# 连接设备
device = connect_device()
if not device:
    exit("无法连接设备")

print(f"已连接: {device.type} @ {device.host}")

# 学习新编码
command_to_learn = "tele_ok"
success = learn_ir_code(device, command_to_learn)
if not success:
    print("学习失败，请检查遥控器和设备状态后重试")

# 发送指令
ensure_ir_codes_file()  # 确保文件存在
with open("ir_codes.json") as f:
    ir_codes = json.load(f)

command = "tele_ok"
if command in ir_codes:
    device.send_data(bytes.fromhex(ir_codes[command]))
    print(f"已发送: {command}")
else:
    print(f"未找到指令: {command}")

# 连接设备
#device = connect_device()
#if not device:
#    exit("无法连接设备")
#print(f"已连接: {device.type} @ {device.host}")
# 学习新编码
command_to_learn = "voice_up"
success = learn_ir_code(device, command_to_learn)
if not success:
    print("学习失败，请检查遥控器和设备状态后重试")

# 发送指令
ensure_ir_codes_file()  # 确保文件存在
with open("ir_codes.json") as f:
    ir_codes = json.load(f)

command = "voice_up"
if command in ir_codes:
    device.send_data(bytes.fromhex(ir_codes[command]))
    print(f"已发送: {command}")
else:
    print(f"未找到指令: {command}")
# 学习新编码
command_to_learn = "voice_down"
success = learn_ir_code(device, command_to_learn)
if not success:
    print("学习失败，请检查遥控器和设备状态后重试")

# 发送指令
ensure_ir_codes_file()  # 确保文件存在
with open("ir_codes.json") as f:
    ir_codes = json.load(f)

command = "voice_down"
if command in ir_codes:
    device.send_data(bytes.fromhex(ir_codes[command]))
    print(f"已发送: {command}")
else:
    print(f"未找到指令: {command}")


