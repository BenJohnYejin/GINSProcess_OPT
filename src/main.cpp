/**
 * @file main.cpp
 * @brief ipos3_opt 命令行入口
 *
 * 读取 ipos3g 的实测数据（DataSensor281_t，256 字节/帧 = 32 个 double），
 * 跑滑窗图优化松组合，并打印统计。
 *
 *   ipos3_opt <xxx.bin> [最大帧数] [输出.nav] [逐帧对比.csv] [选项...]
 *
 * 帧格式（与 ipos3g_cmake/src/nav.cpp 一致）：
 *   0 t | 1..3 wm[3](角增量 rad) | 4..6 vm[3](速度增量 m/s) | 7 dS(里程)
 *   8..10 posgps[3] {lat,lon,h} | 11 flagGNSS | 12..14 vngps[3] {vE,vN,vU}
 *   15 satnum | 16 baseline | 17 yaw(双天线航向 rad) | 18 yawrms
 *   19..21 pos610 | 22..24 att610 | 25..27 vn610 | 28 hoop | 29..31 posstd[3]
 *
 * 选项（都可以不写）：
 *   --yaw-offset=<deg>   双天线航向 -> 内部 RFU/ENU 姿态的固定偏置，默认 0；
 *                        现场数据常见的参考轴差异是 ±90°
 *   --ant-mode=<m>       双天线基线安装方式：FB_B(默认) / FB_F / LR_L / LR_R / ONE
 *                        （与 ipos3g 的 Ant_Mode_* 一致：LR_L 内部 yaw 再 +90°，
 *                          LR_R 再 -90°，FB_F 再 180°；见原工程 GNSS2Ali）
 *   --no-init-yaw        初始姿态不用双天线航向（默认用）
 *   --whiten[=0|1]       预积分残差按协方差白化，默认 1
 *   --bias-jac           残差里带零偏一阶雅可比（零偏可估，但慢 5 倍，默认关）
 *   --vel                打开 GNSS 速度因子
 *   --yaw                打开双天线航向因子
 *   --static             打开静止零速（ZUPT）因子
 *   --gap=<t0>:<t1>      模拟 GNSS 中断（相对首帧的秒数），考核航位推算
 *   --kf=<n>             滑窗关键帧上限，默认 15
 *   --iter=<n>           单次求解最大迭代次数，默认 50
 *   --no-fix-first       不固定窗口首帧 pose（默认固定，做 gauge fix）
 *
 * 约定（按现场确认）：
 *   - IMU 轴系为"右前上"(RFU)，与 ENU 导航系在 yaw=0 时重合；
 *   - 初始姿态直接取水平 + 双天线航向（可加 --yaw-offset），不做双矢量定姿；
 *   - 本批数据无里程计（dS 恒为 99 = 异常标志），不参与解算；
 *   - flagGNSS 非零但 posgps 全零的记录是无效观测，必须剔除。
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "app_interface.h"   /* 对外只需要这一个头（它内部引入 alg_lib.h） */


#define CC180C360(yaw)  ( (yaw)>0.0 ? (_2PI-(yaw)) : -(yaw) )   // counter-clockwise +-180deg -> clockwise 0~360deg for yaw
#define C360CC180(yaw)  ( (yaw)>=PI ? (_2PI-(yaw)) : -(yaw) )   // clockwise 0~360deg -> counter-clockwise +-180deg for yaw


static void printUsage(const char *exe) {
    std::printf("用法: %s <xxx.bin> [最大帧数] [输出.nav] [逐帧对比.csv] [选项...]\n", exe);
    std::printf("选项: --ant-mode=<FB_B|FB_F|LR_L|LR_R|ONE> --yaw-offset=<deg> --no-init-yaw "
                "--whiten[=0|1] --bias-jac --vel --yaw --static "
                "--gap=<t0>:<t1> --kf=<n> --iter=<n> --no-fix-first\n");
}

static bool parseAntMode(const std::string &s, int &mode) {
    if (s == "FB_B" || s == "0") { mode = ANT_MODE_FB_B; return true; }
    if (s == "FB_F" || s == "1") { mode = ANT_MODE_FB_F; return true; }
    if (s == "LR_L" || s == "2") { mode = ANT_MODE_LR_L; return true; }
    if (s == "LR_R" || s == "3") { mode = ANT_MODE_LR_R; return true; }
    if (s == "ONE"  || s == "4") { mode = ANT_MODE_ONE;  return true; }
    return false;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string bin_path, nav_out, cmp_out;
    size_t max_frames   = 0;
    int    n_positional = 0;

    RunnerOptions opt;
    opt.kf_dt         = 1.0;
    opt.max_keyframes = 15;
    opt.use_gnss_vel  = true;
    opt.use_gnss_yaw  = true;
    opt.use_static    = true;
    opt.whiten_preint = false;   /* 见 alg_lib.h：协方差未标定前不白化 */

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--", 0) == 0) {
            const size_t eq = a.find('=');
            const std::string key = (eq == std::string::npos) ? a : a.substr(0, eq);
            const std::string val = (eq == std::string::npos) ? "" : a.substr(eq + 1);
            if (key == "--yaw-offset") {
                opt.yaw_offset = std::atof(val.c_str()) * DEG;
            } else if (key == "--ant-mode") {
                if (!parseAntMode(val, opt.ant_mode)) {
                    std::fprintf(stderr, "未知天线安装方式: %s\n", val.c_str());
                    return 1;
                }
            } else if (key == "--no-init-yaw") {
                opt.init_yaw_from_gnss = false;
            } else if (key == "--whiten") {
                opt.whiten_preint = (val.empty() || std::atof(val.c_str()) != 0.0);
            } else if (key == "--bias-jac") {
                opt.bias_jac = true;
            } else if (key == "--vel") {
                opt.use_gnss_vel = true;
            } else if (key == "--yaw") {
                opt.use_gnss_yaw = true;
            } else if (key == "--static") {
                opt.use_static = true;
            } else if (key == "--no-fix-first") {
                opt.fix_first_pose = false;
            } else if (key == "--kf") {
                opt.max_keyframes = std::atoi(val.c_str());
            } else if (key == "--iter") {
                setenv("IPOS3_MAX_ITER", val.c_str(), 1);
            } else if (key == "--gap") {
                const size_t colon = val.find(':');
                if (colon != std::string::npos) {
                    opt.gnss_gap_t0 = std::atof(val.substr(0, colon).c_str());
                    opt.gnss_gap_t1 = std::atof(val.substr(colon + 1).c_str());
                }
            } else {
                std::fprintf(stderr, "未知选项: %s\n", a.c_str());
                printUsage(argv[0]);
                return 1;
            }
            continue;
        }
        switch (n_positional++) {
            case 0: bin_path   = a; break;
            case 1: max_frames = static_cast<size_t>(std::atol(a.c_str())); break;
            case 2: nav_out    = a; break;
            case 3: cmp_out    = a; break;
            default:
                std::fprintf(stderr, "多余的参数: %s\n", a.c_str());
                printUsage(argv[0]);
                return 1;
        }
    }
    opt.compare_csv_path = cmp_out;

    /* ---- 按 max_frames 截断读入，直接以 vector 交给 runner（不再落临时文件） ---- */
    std::vector<DataSensor281_t> raw;
    if (readSensorFile(bin_path, raw, max_frames) == 0) {
        std::fprintf(stderr, "读取失败\n");
        return 2;
    }

    RunnerStats st = runRealData(raw, opt, nav_out);

    std::printf("\n===== 汇总 =====\n");
    std::printf("  帧数        : %d\n", st.n_frames);
    std::printf("  累计关键帧  : %d\n", st.n_kf_total);
    std::printf("  窗口残留    : %d\n", st.n_kf);
    std::printf("  GNSS 有效   : %d\n", st.n_gnss_used);
    std::printf("  静止帧      : %d\n", st.n_static_frames);
    std::printf("  时长        : %.1f s\n", st.total_time);
    std::printf("  逐帧对比条数: %zu\n", st.per_kf.size());

    std::printf("\n  ---- GNSS 位置对比 ----\n");
    std::printf("  初值 rms   : %.3f m   (max %.3f, P95 %.3f)\n",
                st.init_rms, st.init_max, st.init_pct[2]);
    std::printf("  优化后 rms : %.3f m   (max %.3f, P95 %.3f)\n",
                st.opt_rms,  st.opt_max,  st.opt_pct[2]);
    std::printf("  改善比     : %.2f×\n", st.improve_ratio);

    return 0;
}
