/**
 * @file app_interface.h
 * @brief 【对外唯一入口】ipos3 滑窗图优化（松组合）的应用接口
 *
 * ============================ 工程结构（4 个文件） ============================
 *
 *   app_interface.h   对外唯一入口。基础常量 + 汇总引入下面三部分内容。
 *   alg_lib.h         算法库声明：数学类型 / 地球模型 / IMU 预积分 /
 *                     状态流形 / 图优化因子 / GraphOptimizer / 数据与入口结构。
 *   alg_lib.cpp       算法库实现（按 alg_lib.h 的分节顺序组织）。
 *   main.cpp          命令行入口：读 bin -> runRealData() -> 打印统计。
 *
 * ============================== 怎么用 ==============================
 *
 *   #include "app_interface.h"          // 只需要这一个头
 *
 *   std::vector<DataSensor281_t> raw;
 *   readSensorFile("xxx.bin", raw);      // 256 B/帧的实测数据
 *
 *   RunnerOptions opt;                   // 滑窗、观测、天线安装方式等开关
 *   RunnerStats   st = runRealData(raw, opt, "out.nav");
 *
 *   想直接操作底层（自己组帧、只做一次优化）时，本头引入之后可直接使用：
 *     Preintegration / State / plus / minus / GraphOptimizer / llt_* / schur_complement
 *   这些名字都来自 alg_lib.h，由本文件在末尾统一引入。
 *
 * ============================ 内部实现分层 ============================
 *
 *   app_interface.h   常量（本文件，独立于其它头，无循环依赖）
 *        |
 *        v
 *   alg_lib.h         全部类型与接口声明
 *        |
 *        v
 *   alg_lib.cpp       实现
 *
 * 说明：alg_lib.h 顶部 #include 本文件取常量；本文件末尾 #include alg_lib.h
 * 把声明带回来。两个头都有 include guard，因此"先引哪个"都能正常展开，
 * 循环包含不会出问题。
 */

#ifndef APPINTERFACE_H
#define APPINTERFACE_H

#include <cmath>
#include <cstdint>

constexpr double const_sqrt(double x, double guess) {
    double r = guess;
    for (int i = 0; i < 60; i++) {
        r = 0.5 * (r + x / r);
    }
    return r;
}

constexpr double PI   = 3.1415926535897932;
constexpr double _2PI = 2.0 * PI;
constexpr double DEG  = PI / 180.0;   /*< 度 -> 弧度 */
constexpr double HUR  = 3600.0;
constexpr double SHUR = 60.0;
constexpr double DPS  = DEG;          /*< deg/s   -> rad/s   */
constexpr double DPH  = DEG / HUR;    /*< deg/hr  -> rad/s   */
constexpr double DPSH = DEG / SHUR;   /*< deg/sqrt(hr) -> rad/sqrt(s) */

constexpr double G0 = 9.7803267714;
constexpr double MG = G0 / 1.0e3;
constexpr double UG = G0 / 1.0e6;

constexpr double RE_WGS84 = 6378137.0;
constexpr double FE_WGS84 = 1.0 / 298.257223563;
constexpr double RE       = 6378137.0;
constexpr double f0_earth = 1.0 / 298.257;
constexpr double wie0     = 7.2921151467e-5;

constexpr double RP = (1.0 - f0_earth) * RE; /*< 极半径 */
constexpr double e_earth = const_sqrt(2.0 * f0_earth - f0_earth * f0_earth, 0.08);
constexpr double e2      = e_earth * e_earth;
constexpr double ep2     = e2 / (1.0 - e2);
constexpr double ep_earth = const_sqrt(ep2, 0.08);

constexpr double EPS   = 2.220446049e-16;
constexpr double INF   = 3.402823466e+30;
constexpr double INFp5 = INF * 0.5;

inline bool isInfSentinel(double v) { return v > 2.0 * INF; }

double range(double val, double minVal, double maxVal);
int sign(double val, double eps = EPS);
double atan2Ex(double y, double x);
inline double asinEx(double x) { return std::asin(range(x, -1.0, 1.0)); }

template <class T>
inline void swapt(T &a, T &b) {
    T c = a;
    a   = b;
    b   = c;
}

double norm(const double *pd, int n);
double norm1(const double *pd, int n);
double normInf(const double *pd, int n);

/* 参数块维数 ---------------------------------------------------------------------*/
constexpr int NUM_POSE      = 7;  /*< p(3) + q(4) */
constexpr int NUM_MIX       = 9;  /*< v(3) + bg(3) + ba(3) */
constexpr int NUM_MIX_ODO   = 10; /*< 再加上里程计比例因子误差 sodo */
constexpr int NUM_STATE     = 15; /*< 预积分残差维数 */
constexpr int NUM_STATE_ODO = 19; /*< 再加上里程位移(3) 与比例因子(1) */
constexpr int NUM_NOISE     = 12; /*< IMU 噪声维数 */
constexpr int NUM_NOISE_ODO = 16; /*< 再加上里程计白噪声(3) 与比例因子随机游走(1) */

/* ============================================================================
 * 对外接口总入口
 * ----------------------------------------------------------------------------
 * 下面引入算法库的全部声明。放在文件末尾是为了让 alg_lib.h 能先拿到本文件
 * 里的常量；两个头都有 guard，先引哪个都不会出错。
 * ==========================================================================*/
#include "alg_lib.h"

#endif  // APPINTERFACE_H
