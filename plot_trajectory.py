"""
plot_trajectory.py

用法：
1. 直接绘制已有轨迹文件
   python plot_trajectory.py

2. 指定目标经纬度，先生成环绕轨迹再绘制
   python plot_trajectory.py --target-lat 23.0 --target-lon 123.8
   可选：--target-name "目标A"（用于目标位置命名；并用于默认轨迹 txt 文件名）
   曲线轨迹：python plot_trajectory.py --target-lat 23.0 --target-lon 123.8 --trajectory-mode curve

依赖：matplotlib
"""

import argparse
import math
import re
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
LONGITUDE_SCALE_EPSILON = 1e-9
AZIMUTH_ROUND_DECIMALS = 10
ANGLE_FORMAT = ".3f"
COORDINATE_FORMAT = ".6f"
ALTITUDE_FORMAT = ".3f"
TIME_FORMAT = ".3f"
CURVE_AZIMUTH_ROUND_DECIMALS = AZIMUTH_ROUND_DECIMALS
MIN_AZIMUTH_SPAN_FOR_LOOP_CLOSURE = 180.0

AZIMUTH_COLUMN_INDEX = 15
LATITUDE_COLUMN_INDEX = 16
LONGITUDE_COLUMN_INDEX = 17
ALTITUDE_COLUMN_INDEX = 18
DEFAULT_INPUT_FILENAME = "trajectory.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="绘制轨迹，或按目标经纬度生成并绘制轨迹。")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILENAME,
        help="要读取的轨迹文件；若同时指定目标经纬度，则生成后从该文件读取。",
    )
    parser.add_argument("--target-lat", type=float, help="目标纬度")
    parser.add_argument("--target-lon", type=float, help="目标经度")
    parser.add_argument("--target-name", help="目标名称（用于文件命名和图中标注）")
    parser.add_argument("--altitude", type=float, default=10000.0, help="飞行器高度（米）")
    parser.add_argument("--pitch", type=float, default=85.0, help="俯仰角（度）")
    parser.add_argument(
        "--trajectory-mode",
        choices=["circle", "curve"],
        default="circle",
        help="轨迹模式：circle 为绕目标环绕，curve 为曲线通过。",
    )
    parser.add_argument("--azimuth-step", type=float, default=10.0, help="方位角步长（度）")
    parser.add_argument("--curve-points", type=int, default=37, help="曲线轨迹采样点数（>=2）")
    parser.add_argument("--curve-span", type=float, default=2000.0, help="曲线跨越长度（米）")
    parser.add_argument(
        "--curve-bulge",
        type=float,
        default=0.0,
        help="曲线横向弯曲幅度（米，允许为0；0表示沿主方向直线穿越目标）",
    )
    parser.add_argument("--curve-bearing", type=float, default=90.0, help="曲线主方向（度，0北90东）")
    parser.add_argument("--curve-duration", type=float, default=100.0, help="曲线总时长（秒，默认100秒；仅 curve 模式生效）")
    parser.add_argument("--curve-peak-altitude", type=float, help="曲线最高点高度（米，默认比基准高度高3000米）")
    parser.add_argument("--time-step", type=float, default=0.1, help="时间步长（秒）")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存图片，不弹出窗口（适合服务器环境）。",
    )
    return parser.parse_args()


def slugify_target_name(target_name):
    return re.sub(r"[^\w-]+", "_", target_name.strip()).strip("_")


def default_output_path_for_target(target_lat, target_lon, target_name, trajectory_mode):
    prefix = "trajectory_curve" if trajectory_mode == "curve" else "trajectory"
    if target_name:
        safe_name = slugify_target_name(target_name)
        if safe_name:
            return Path(f"{prefix}_{safe_name}.txt")
    lat_tag = f"{target_lat:.6f}".replace("-", "m").replace(".", "p")
    lon_tag = f"{target_lon:.6f}".replace("-", "m").replace(".", "p")
    return Path(f"{prefix}_lat{lat_tag}_lon{lon_tag}.txt")


def calculate_azimuth_deg(delta_north, delta_east):
    """由北向与东向位移分量计算方位角（度，范围 [0, 360)）。"""
    return math.degrees(math.atan2(delta_east, delta_north)) % 360.0


def generate_trajectory_file(
    output_path,
    target_lat,
    target_lon,
    altitude=10000.0,
    pitch_deg=85.0,
    azimuth_step=10.0,
    time_step=0.1,
):
    """生成环绕目标的圆形轨迹，并按19列制表符格式写入txt。"""
    if altitude <= 0:
        raise ValueError("altitude 必须为正数。")
    if not (0 < pitch_deg < 90):
        raise ValueError("pitch 必须在 0 到 90 度之间（不含端点），以生成当前这种有非零水平半径的环绕轨迹。")
    if azimuth_step <= 0:
        raise ValueError("azimuth-step 必须大于 0。")
    if time_step <= 0:
        raise ValueError("time-step 必须大于 0。")

    horizontal_radius = altitude * math.tan(math.radians(90.0 - pitch_deg))
    lon_scale = METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(target_lat))
    if abs(lon_scale) < LONGITUDE_SCALE_EPSILON:
        raise ValueError("目标纬度过于接近极点，无法稳定换算经度偏移。")

    rows = [HEADER]
    index = 0
    azimuth = 0.0
    # 在 360° 前停止，避免浮点累积误差让最后一次循环重复到起点。
    while azimuth < 360.0 - AZIMUTH_LOOP_EPSILON:
        azimuth_rad = math.radians(azimuth)
        delta_north = horizontal_radius * math.cos(azimuth_rad)
        delta_east = horizontal_radius * math.sin(azimuth_rad)

        lat = target_lat + delta_north / METERS_PER_DEGREE_LATITUDE
        lon = target_lon + delta_east / lon_scale
        time_value = index * time_step

        row = [
            f"{time_value:{TIME_FORMAT}}",
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
            f"{pitch_deg:{ANGLE_FORMAT}}",
            f"{azimuth:{ANGLE_FORMAT}}",
            f"{lat:{COORDINATE_FORMAT}}",
            f"{lon:{COORDINATE_FORMAT}}",
            f"{altitude:{ALTITUDE_FORMAT}}",
        ]
        rows.append("\t".join(row))

        index += 1
        # 对累积方位角做有限精度舍入，避免浮点误差导致 360° 附近多出或少掉一个点。
        azimuth = round(index * azimuth_step, AZIMUTH_ROUND_DECIMALS)

    output_path = Path(output_path)
    output_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"轨迹文件已生成：{output_path}")


def generate_curve_trajectory_file(
    output_path,
    target_lat,
    target_lon,
    altitude=10000.0,
    pitch_deg=85.0,
    point_count=37,
    span_meters=2000.0,
    bulge_meters=0.0,
    bearing_deg=90.0,
    duration_seconds=100.0,
    peak_altitude=None,
):
    """生成先升后降并越过目标上方的曲线轨迹，并按19列制表符格式写入txt。"""
    if altitude <= 0:
        raise ValueError("altitude 必须为正数。")
    if not (0 < pitch_deg < 90):
        raise ValueError("pitch 必须在 0 到 90 度之间（不含端点）。")
    if point_count < 2:
        raise ValueError("curve-points 必须大于等于 2。")
    if span_meters <= 0:
        raise ValueError("curve-span 必须大于 0。")
    if bulge_meters < 0:
        raise ValueError("curve-bulge 不能为负数。")
    if duration_seconds <= 0:
        raise ValueError("curve-duration 必须大于 0。")

    lon_scale = METERS_PER_DEGREE_LATITUDE * math.cos(math.radians(target_lat))
    if abs(lon_scale) < LONGITUDE_SCALE_EPSILON:
        raise ValueError("目标纬度过于接近极点，无法稳定换算经度偏移。")
    if peak_altitude is None:
        peak_altitude = altitude + 3000.0
    if peak_altitude <= altitude:
        raise ValueError("curve-peak-altitude 必须大于 altitude，才能形成先升后降曲线。")

    heading_rad = math.radians(bearing_deg)
    point_offsets = []
    for index in range(point_count):
        progress = index / (point_count - 1)
        longitudinal_offset = (progress - 0.5) * span_meters
        # sin(2πp) 在 p=0/0.5/1 时为 0：起点、中点、终点横向偏移为 0（默认 bulge=0 时全程为 0）。
        lateral_offset = bulge_meters * math.sin(2.0 * math.pi * progress)
        delta_north = longitudinal_offset * math.cos(heading_rad) - lateral_offset * math.sin(heading_rad)
        delta_east = longitudinal_offset * math.sin(heading_rad) + lateral_offset * math.cos(heading_rad)
        point_offsets.append((delta_north, delta_east))

    rows = [HEADER]
    for index, (delta_north, delta_east) in enumerate(point_offsets):
        progress = index / (point_count - 1)
        if index == 0:
            next_north, next_east = point_offsets[index + 1]
            motion_north = next_north - delta_north
            motion_east = next_east - delta_east
        elif index == point_count - 1:
            prev_north, prev_east = point_offsets[index - 1]
            motion_north = delta_north - prev_north
            motion_east = delta_east - prev_east
        else:
            prev_north, prev_east = point_offsets[index - 1]
            next_north, next_east = point_offsets[index + 1]
            motion_north = next_north - prev_north
            motion_east = next_east - prev_east

        lat = target_lat + delta_north / METERS_PER_DEGREE_LATITUDE
        lon = target_lon + delta_east / lon_scale
        # sin(πp) 在 p=0 和 p=1 为 0，在 p=0.5 为 1：前半程爬升、后半程下降，并在中段达到最高点。
        altitude_value = altitude + (peak_altitude - altitude) * math.sin(math.pi * progress)
        azimuth = round(calculate_azimuth_deg(motion_north, motion_east), CURVE_AZIMUTH_ROUND_DECIMALS)
        time_value = progress * duration_seconds
        row = [
            f"{time_value:{TIME_FORMAT}}",
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
            f"{pitch_deg:{ANGLE_FORMAT}}",
            f"{azimuth:{ANGLE_FORMAT}}",
            f"{lat:{COORDINATE_FORMAT}}",
            f"{lon:{COORDINATE_FORMAT}}",
            f"{altitude_value:{ALTITUDE_FORMAT}}",
        ]
        rows.append("\t".join(row))

    output_path = Path(output_path)
    output_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"曲线轨迹文件已生成：{output_path}")


def load_trajectory(txt_file):
    """读取19列制表符轨迹txt，返回 path/times/lats/lons/alts/azimuths 字典。"""
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
            alts.append(float(columns[ALTITUDE_COLUMN_INDEX]))
            azimuths.append(float(columns[AZIMUTH_COLUMN_INDEX]))
            lats.append(float(columns[LATITUDE_COLUMN_INDEX]))
            lons.append(float(columns[LONGITUDE_COLUMN_INDEX]))

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


def save_plots(data, show_plots=True, target_label_base="目标物"):
    """保存轨迹3D图与俯视图；大跨度方位角轨迹会按闭环方式绘制。"""
    lats = data["lats"]
    lons = data["lons"]
    alts = data["alts"]
    azimuths = data["azimuths"]
    prefix = data["path"].with_suffix("")
    output_3d = prefix.with_name(prefix.name + "_3d.png")
    output_top = prefix.with_name(prefix.name + "_top.png")
    azimuth_span = max(azimuths) - min(azimuths) if len(azimuths) > 1 else 0.0
    azimuth_steps = [abs(current - previous) for previous, current in zip(azimuths, azimuths[1:]) if abs(current - previous) > 0]
    representative_step = min(azimuth_steps) if azimuth_steps else 10.0
    loop_visualization_threshold = max(
        MIN_AZIMUTH_SPAN_FOR_LOOP_CLOSURE,
        360.0 - 2.0 * representative_step,
    )
    should_use_loop_visualization = azimuth_span >= loop_visualization_threshold
    midpoint_label = "中点位置"
    plot_title = "飞行器绕目标物飞行一圈轨迹" if should_use_loop_visualization else "飞行器曲线轨迹"
    top_plot_title = "飞行轨迹俯视图（经纬度平面）"

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    path_lons = lons + [lons[0]] if should_use_loop_visualization else lons
    path_lats = lats + [lats[0]] if should_use_loop_visualization else lats
    path_alts = alts + [alts[0]] if should_use_loop_visualization else alts

    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(
        path_lons,
        path_lats,
        path_alts,
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
        label=midpoint_label,
    )
    axis.scatter(
        [center_lon],
        [center_lat],
        [0],
        color="orange",
        s=120,
        marker="*",
        zorder=5,
        label=f"{target_label_base}",
    )

    for lon, lat, alt, azimuth in zip(lons, lats, alts, azimuths):
        axis.text(lon, lat, alt + 30, f"{int(round(azimuth))}°", fontsize=6, ha="center", color="gray")

    axis.set_xlabel("经度 (°E)")
    axis.set_ylabel("纬度 (°N)")
    axis.set_zlabel("高度 (m)")
    axis.set_title(plot_title)
    axis.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_3d, dpi=150)
    print(f"3D 图已保存至 {output_3d}")

    figure2, axis2 = plt.subplots(figsize=(7, 7))
    axis2.plot(path_lons, path_lats, "b-o", markersize=5, label="飞行轨迹")
    axis2.scatter(lons[0], lats[0], color="green", s=80, zorder=5, label="起点")
    axis2.scatter(lons[half_index], lats[half_index], color="red", s=80, zorder=5, label=midpoint_label)
    axis2.scatter(
        center_lon,
        center_lat,
        color="orange",
        s=120,
        marker="*",
        zorder=5,
        label=f"{target_label_base}中心",
    )

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
    axis2.set_title(top_plot_title)
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
        output_path = Path(args.input)
        if args.input == DEFAULT_INPUT_FILENAME:
            output_path = default_output_path_for_target(
                args.target_lat,
                args.target_lon,
                args.target_name,
                args.trajectory_mode,
            )
        if args.trajectory_mode == "curve":
            generate_curve_trajectory_file(
                output_path=output_path,
                target_lat=args.target_lat,
                target_lon=args.target_lon,
                altitude=args.altitude,
                pitch_deg=args.pitch,
                point_count=args.curve_points,
                span_meters=args.curve_span,
                bulge_meters=args.curve_bulge,
                bearing_deg=args.curve_bearing,
                duration_seconds=args.curve_duration,
                peak_altitude=args.curve_peak_altitude,
            )
        else:
            generate_trajectory_file(
                output_path=output_path,
                target_lat=args.target_lat,
                target_lon=args.target_lon,
                altitude=args.altitude,
                pitch_deg=args.pitch,
                azimuth_step=args.azimuth_step,
                time_step=args.time_step,
            )
    else:
        output_path = Path(args.input)

    trajectory_data = load_trajectory(output_path)
    target_label = f"目标物:{args.target_name}" if args.target_name else "目标物"
    save_plots(trajectory_data, show_plots=not args.no_show, target_label_base=target_label)


if __name__ == "__main__":
    main()
