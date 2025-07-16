import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Vehicle import Target

class TargetRefresher:
    def __init__(self, area_size=9260, start_id=3):
        self.area_size = area_size
        self.current_id = start_id
        self.target_refresh_enabled = False
    
    def isValid(self, targets):
        for tid, target in list(targets.items()):  # 遍历副本
            x, y = target.position
            if not (0 <= x <= self.area_size and 0 <= y <= self.area_size):
                del targets[tid]
                self.target_refresh_enabled = True
                
    def isCaptured(self, targets, captured):
        if captured != None:
            for target_id in targets.keys():
                if target_id in captured:
                    self.target_refresh_enabled = True
                
    def refresh(self, targets, t):
        if self.target_refresh_enabled == True:
            NewTarget = Target(self.current_id, t)
            targets['self.current_id'] = NewTarget
            self.current_id += 1
            self.target_refresh_enabled = False
            
            
        