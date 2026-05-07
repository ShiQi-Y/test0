"""
plot_trajectory.py
读取 trajectory.txt，绘制飞行器绕目标物飞行一圈的 3D 轨迹图。
依赖：matplotlib
"""

import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

TXT_FILE = "trajectory.txt"

# ---------- 读取数据 ----------
times, lats, lons, alts, azimuths = [], [], [], [], []

with open(TXT_FILE, encoding="utf-8") as f:
    for lineno, line in enumerate(f):
        line = line.strip()
        if lineno == 0 or not line:   # 跳过表头
            continue
        cols = line.split("\t")
        times.append(float(cols[0]))
        alts.append(float(cols[18]))
        azimuths.append(float(cols[15]))
        lats.append(float(cols[16]))
        lons.append(float(cols[17]))

# ---------- 3D 轨迹图 ----------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# 绘制轨迹（闭合，首尾相连）
x = lons + [lons[0]]
y = lats + [lats[0]]
z = alts + [alts[0]]

ax.plot(x, y, z, "b-o", markersize=4, linewidth=1.5, label="飞行轨迹")

# 标注起始点
ax.scatter([lons[0]], [lats[0]], [alts[0]], color="green", s=80, zorder=5, label="起点 (Az=0°)")
ax.scatter([lons[18]], [lats[18]], [alts[18]], color="red",   s=80, zorder=5, label="半圈 (Az=180°)")

# 目标物（中心点）投影到地面
center_lat = sum(lats) / len(lats)
center_lon = sum(lons) / len(lons)
ax.scatter([center_lon], [center_lat], [0], color="orange", s=120,
           marker="*", zorder=5, label="目标物 (地面)")

# 标注每个点的方位角
for i in range(len(lons)):
    ax.text(lons[i], lats[i], alts[i] + 30,
            f"{int(azimuths[i])}°", fontsize=6, ha="center", color="gray")

ax.set_xlabel("经度 (°E)")
ax.set_ylabel("纬度 (°N)")
ax.set_zlabel("高度 (m)")
ax.set_title("飞行器绕目标物飞行一圈轨迹\n(俯仰角=85°，方位角步长=10°，时间步长=0.1s)")
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig("trajectory_3d.png", dpi=150)
print("3D 图已保存至 trajectory_3d.png")
plt.show()

# ---------- 俯视图（经纬度平面） ----------
fig2, ax2 = plt.subplots(figsize=(7, 7))
ax2.plot(lons + [lons[0]], lats + [lats[0]], "b-o", markersize=5, label="飞行轨迹")
ax2.scatter(lons[0], lats[0], color="green", s=80, zorder=5, label="起点 (Az=0°)")
ax2.scatter(lons[18], lats[18], color="red", s=80, zorder=5, label="半圈 (Az=180°)")
ax2.scatter(center_lon, center_lat, color="orange", s=120, marker="*",
            zorder=5, label="目标物中心")

# 标注方位角
for i in range(len(lons)):
    ax2.annotate(f"{int(azimuths[i])}°",
                 (lons[i], lats[i]),
                 textcoords="offset points", xytext=(4, 4),
                 fontsize=7, color="gray")

ax2.set_xlabel("经度 (°E)")
ax2.set_ylabel("纬度 (°N)")
ax2.set_title("飞行轨迹俯视图（经纬度平面）")
ax2.legend()
ax2.set_aspect("equal")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("trajectory_top.png", dpi=150)
print("俯视图已保存至 trajectory_top.png")
plt.show()
