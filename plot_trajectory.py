"""
plot_trajectory.py

用法：
1. 直接绘制已有轨迹文件
   python plot_trajectory.py

2. 指定目标经纬度，先生成环绕轨迹再绘制
   python plot_trajectory.py --target-lat 23.0 --target-lon 123.8

依赖：matplotlib
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HEADER = "\t".join(
    [
        "时间(s)",
        "位置x",
        "位置y",
        "位置z",
        "phi",
        "psi",
        "gm",
        "q0",
        "q1",
        "q2",
        "q3",
        "速度x",
        "速度y",
        "速度z",
        "俯仰角",
        "方位角",
        "纬度",
        "经度",
        "高度",
    ]
)
METERS_PER_DEGREE_LATITUDE = 111000.0
AZIMUTH_LOOP_EPSILON = 1e-9
AZIMUTH_ROUND_DECIMALS = 10


def parse_args():
    parser = argparse.ArgumentParser(description="绘制轨迹，或按目标经纬度生成并绘制轨迹。")
    parser.add_argument(
        "--input",
        default="trajectory.txt",
        help="要读取的轨迹文件；若同时指定目标经纬度，则生成后从该文件读取。",
    )
    parser.add_argument("--target-lat", type=float, help="目标纬度")
    parser.add_argument("--target-lon", type=float, help="目标经度")
    parser.add_argument("--altitude", type=float, default=10000.0, help="飞行器高度（米）")
    parser.add_argument("--pitch", type=float, default=85.0, help="俯仰角（度）")
    parser.add_argument("--azimuth-step", type=float, default=10.0, help="方位角步长（度）")
    parser.add_argument("--time-step", type=float, default=0.1, help="时间步长（秒）")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存图片，不弹出窗口（适合服务器环境）。",
    )
    return parser.parse_args()


def generate_trajectory_file(
    output_path,
    target_lat,
    target_lon,
    altitude=10000.0,
    pitch_deg=85.0,
    azimuth_step=10.0,
    time_step=0.1,
):
    if altitude <= 0:
        raise ValueError("altitude 必须为正数。")
    if not (0 < pitch_deg < 90):
        raise ValueError("pitch 必须在 0 到 90 度之间（不含端点），以确保水平半径计算有效。")
    if azimuth_step <= 0:
        raise ValueError("azimuth-step 必须大于 0。")
    if time_step <= 0:
        raise ValueError("time-step 必须大于 0。")

    horizontal_radius = altitude * math.tan(math.radians(90.0 - pitch_deg))
    lon_scale = METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(target_lat))
    if abs(lon_scale) < 1e-9:
        raise ValueError("目标纬度过于接近极点，无法稳定换算经度偏移。")

    rows = [HEADER]
    index = 0
    azimuth = 0.0
    while azimuth < 360.0 - AZIMUTH_LOOP_EPSILON:
        azimuth_rad = math.radians(azimuth)
        delta_north = horizontal_radius * math.cos(azimuth_rad)
        delta_east = horizontal_radius * math.sin(azimuth_rad)

        lat = target_lat + delta_north / METERS_PER_DEGREE_LATITUDE
        lon = target_lon + delta_east / lon_scale
        time_value = index * time_step

        row = [
            f"{time_value:.3f}",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            f"{pitch_deg:.3f}",
            f"{azimuth:.3f}",
            f"{lat:.6f}",
            f"{lon:.6f}",
            f"{altitude:.3f}",
        ]
        rows.append("\t".join(row))

        index += 1
        # 对累积方位角做有限精度舍入，避免浮点误差导致 360° 附近多出或少掉一个点。
        azimuth = round(index * azimuth_step, AZIMUTH_ROUND_DECIMALS)

    output_path = Path(output_path)
    output_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"轨迹文件已生成：{output_path}")


def load_trajectory(txt_file):
    txt_path = Path(txt_file)
    if not txt_path.exists():
        raise FileNotFoundError(f"未找到轨迹文件：{txt_path}")

    times, lats, lons, alts, azimuths = [], [], [], [], []
    with txt_path.open(encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj):
            line = line.strip()
            if line_number == 0 or not line:
                continue
            columns = line.split("\t")
            if len(columns) < 19:
                raise ValueError(
                    f"文件 {txt_path} 第 {line_number + 1} 行列数不足：期望至少 19 列，实际 {len(columns)} 列。"
                )
            times.append(float(columns[0]))
            alts.append(float(columns[18]))
            azimuths.append(float(columns[15]))
            lats.append(float(columns[16]))
            lons.append(float(columns[17]))

    if not lats:
        raise ValueError(f"轨迹文件为空：{txt_path}")

    return {
        "path": txt_path,
        "times": times,
        "lats": lats,
        "lons": lons,
        "alts": alts,
        "azimuths": azimuths,
    }


def save_plots(data, show_plots=True):
    lats = data["lats"]
    lons = data["lons"]
    alts = data["alts"]
    azimuths = data["azimuths"]
    prefix = data["path"].with_suffix("")
    output_3d = prefix.with_name(prefix.name + "_3d.png")
    output_top = prefix.with_name(prefix.name + "_top.png")

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(
        lons + [lons[0]],
        lats + [lats[0]],
        alts + [alts[0]],
        "b-o",
        markersize=4,
        linewidth=1.5,
        label="飞行轨迹",
    )
    axis.scatter([lons[0]], [lats[0]], [alts[0]], color="green", s=80, zorder=5, label="起点")

    half_index = len(lons) // 2
    axis.scatter(
        [lons[half_index]],
        [lats[half_index]],
        [alts[half_index]],
        color="red",
        s=80,
        zorder=5,
        label="半圈位置",
    )
    axis.scatter([center_lon], [center_lat], [0], color="orange", s=120, marker="*", zorder=5, label="目标物")

    for lon, lat, alt, azimuth in zip(lons, lats, alts, azimuths):
        axis.text(lon, lat, alt + 30, f"{int(round(azimuth))}°", fontsize=6, ha="center", color="gray")

    axis.set_xlabel("经度 (°E)")
    axis.set_ylabel("纬度 (°N)")
    axis.set_zlabel("高度 (m)")
    axis.set_title("飞行器绕目标物飞行一圈轨迹")
    axis.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_3d, dpi=150)
    print(f"3D 图已保存至 {output_3d}")

    figure2, axis2 = plt.subplots(figsize=(7, 7))
    axis2.plot(lons + [lons[0]], lats + [lats[0]], "b-o", markersize=5, label="飞行轨迹")
    axis2.scatter(lons[0], lats[0], color="green", s=80, zorder=5, label="起点")
    axis2.scatter(lons[half_index], lats[half_index], color="red", s=80, zorder=5, label="半圈位置")
    axis2.scatter(center_lon, center_lat, color="orange", s=120, marker="*", zorder=5, label="目标物中心")

    for lon, lat, azimuth in zip(lons, lats, azimuths):
        axis2.annotate(
            f"{int(round(azimuth))}°",
            (lon, lat),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="gray",
        )

    axis2.set_xlabel("经度 (°E)")
    axis2.set_ylabel("纬度 (°N)")
    axis2.set_title("飞行轨迹俯视图（经纬度平面）")
    axis2.legend()
    axis2.set_aspect("equal")
    axis2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_top, dpi=150)
    print(f"俯视图已保存至 {output_top}")

    if show_plots:
        plt.show()
    else:
        plt.close("all")


def main():
    args = parse_args()
    should_generate = args.target_lat is not None or args.target_lon is not None
    if should_generate:
        if args.target_lat is None or args.target_lon is None:
            raise ValueError("如果要生成轨迹，必须同时提供 --target-lat 和 --target-lon。")
        generate_trajectory_file(
            output_path=args.input,
            target_lat=args.target_lat,
            target_lon=args.target_lon,
            altitude=args.altitude,
            pitch_deg=args.pitch,
            azimuth_step=args.azimuth_step,
            time_step=args.time_step,
        )

    trajectory_data = load_trajectory(args.input)
    save_plots(trajectory_data, show_plots=not args.no_show)


if __name__ == "__main__":
    main()
