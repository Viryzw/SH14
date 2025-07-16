import time
from Utils.Manager import Manager  
from Utils.Refresher import TargetRefresher

# 初始化系统
manager = Manager()
refresher = TargetRefresher()

# UAV 和 USV 初始配置
uavs = [
    ["1", [0, 5630], 0],
    ["2", [0, 3630], 0]
]
usvs = [
    ["1", [0, 6130], 0],
    ["2", [0, 5130], 0],
    ["3", [0, 4130], 0],
    ["4", [0, 3130], 0]
]
targets = [
    ["1"],
    ["2"]
]

# 初始化对象
manager.init_objects(uavs, usvs, targets, 0)

# 主仿真循环
max_step = 144000
log_interval = 1
capture_count = 0

for step in range(max_step):
    # 控制信息（模拟简单控制）
    controls = [
        ["uav", "1", 50, 0],
        ["uav", "2", 50, 0],
        ["usv", "1", 50, 0],
        ["usv", "2", 50, 0],
        ["usv", "3", 50, 0],
        ["usv", "4", 50, 0],
    ]

    # 所有目标以固定速度前进
    for tid in list(manager.targets.keys()):
        controls.append(["target", tid, 20, 0])

    # 执行更新
    manager.update(controls, t=step)  # 时间步长0.05s

    # 自动刷新目标
    refresher.isValid(manager.targets)
    refresher.isCaptured(manager.targets)
    refresher.refresh(manager.targets, step)

    # 每 log_interval 步打印一次系统状态
    if step % log_interval == 0:
        print(f"\n=== Step {step} 状态 ===")
        for uid in ['1', '2']:
            detected = manager.get_detected('uav', uid)
            print(f"UAV {uid} 探测到目标: {detected}")
        # for uid in ['1', '2', '3', '4']:
        #     captured = manager.get_captured('usv', uid)
        #     print(f"USV {uid} 捕获的目标: {captured}")
        captured = manager.get_captured('usv')
        print(f"USV捕获的目标: {captured}")
        # print(f"当前剩余目标数: {len(manager.targets)}")
        for key, value in manager.targets.items():
            print(f"当前目标状态--key: {key}, value: {value.position}")
    print(f"探测时间记录: {manager.time1}")
    print(f"捕获时间记录: {manager.time2}")
    
    time.sleep(0.05)