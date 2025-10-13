#!/usr/bin/env python3
"""
Quick evaluator for ebot_nav CSV log.
Run after finishing a trial. It prints whether P1,P2,P3 were achieved within tolerances.
"""

import pandas as pd
import math
import glob
import os

# find most recent ebot_nav log file
files = sorted(glob.glob("ebot_nav*.csv"), key=os.path.getmtime)
if not files:
    print("No ebot_nav log file found in current directory.")
    exit(1)
logfile = files[-1]
print("Checking log:", logfile)

df = pd.read_csv(logfile)

waypoints = [
    (-1.53, -1.95, 1.57),
    (0.13,  1.24,  0.00),
    (0.38, -3.32, -1.57)
]

pos_tol = 0.3
yaw_tol = math.radians(10)

def reached_goal(df, gx, gy, gyaw):
    # compute distance to goal for each logged row
    d = ((df['x'] - gx)**2 + (df['y'] - gy)**2)**0.5
    # compute yaw difference safely, considering wrapping
    yaw_diff = df['yaw'].apply(lambda y: abs(math.atan2(math.sin(gyaw - y), math.cos(gyaw - y))))
    # check any timestamp where both within tolerance
    ok = ((d <= pos_tol) & (yaw_diff <= yaw_tol)).any()
    return ok

all_ok = True
for i, (gx, gy, gyaw) in enumerate(waypoints, start=1):
    ok = reached_goal(df, gx, gy, gyaw)
    print(f"P{i}: {'✅ PASS' if ok else '❌ FAIL'}")
    all_ok = all_ok and ok

min_front = df['min_front_range'].min()
print(f"Minimum front range observed during run: {min_front:.3f} m")

if all_ok:
    print("\nAll waypoints reached: ✅ Ready to run autoeval & record video.")
else:
    print("\nSome waypoints not reached. Tweak controller/gains or check collisions/log for details.")
