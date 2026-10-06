/**
 * @file alg_lib.h
 * @brief 算法库声明（数学 / 地球模型 / IMU 预积分 / 图优化）
 *
 * 【不要直接引用本文件】对外只需要 #include "app_interface.h"，
 * 由它在末尾统一引入本文件；两个头都有 guard，顺序无所谓。
 *
 * 分节（与 alg_lib.cpp 的实现顺序一一对应）：
 *   0. 基础常量 ................ app_interface.h
 *   1. 基础类型 ................ Vector3d / Matrix3d / Quaterniond（后续迁移）
           *                              动态向量/矩阵已统一为 Eigen
 *   2. 标量 & 矢量工具 .......... range / diffYaw / norm / askew ...
 *   3. 姿态与旋转 .............. a2qua / rv2q / q2mat / m2att ...
 *   4. 线性代数 ................ inv / llt_L / llt_sqrtinv / pinv_sym
 *   5. 地球模型与坐标系 ........ earth / LocalFrame / xyz2blh ...
 *   6. IMU 与预积分 ............ IMU / Preintegration / ImuMeas
 *   7. 状态与流形 .............. State / plus / minus / applyDelta
 *   8. 边缘化 .................. accumulate_normal_equations / schur_complement
 *                              / MarginalizationInfo
 *   9. 图优化 .................. PreintResidual / Gnss*Residual /
 *                              MarginalizationPriorFactor / GraphOptimizer
 *  10. 数据与运行入口 .......... DataSensor281_t / RunnerOptions / RunnerStats
 *                              / readSensorFile / runRealData
 */

#ifndef IPOS_ALIG_H
#define IPOS_ALIG_H

/* 基础常量（PI / DEG / G0 / NUM_* ...）：本文件用，也在 app_interface.h 里
 * 统一对外暴露。末尾由 app_interface.h 反向引入本文件，完成"只引一个头"。 */

#include <array>
#include <algorithm>
#include <set>
#include <random>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <deque>
#include <memory>
#include <vector>
#include <map>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <ceres/ceres.h>
#include <string>
#include <cctype>
#include <fstream>
#include <limits>
#include <Eigen/Dense>

#include <filesystem>                              // std::filesystem
#include <opencv2/core.hpp>                        // cv::Mat / Point2f / TermCriteria / Scalar
#include <opencv2/imgproc.hpp>                     // goodFeaturesToTrack / circle
#include <opencv2/features2d.hpp>                  // (备用)
#include <opencv2/video/tracking.hpp>              // calcOpticalFlowPyrLK / OPTFLOW_USE_INITIAL_FLOW
#include <opencv2/calib3d.hpp>                     // findFundamentalMat / FM_RANSAC / undistortPoints
#include <opencv2/imgcodecs.hpp>                   // imread / IMREAD_GRAYSCALE

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

/* ============================ 多频 GNSS 常量 ============================ */
constexpr int    MAX_FREQ = 3;
constexpr double CLIGHT   = 299792458.0;

struct FreqInfo {
    double freq_hz;
    double lambda;
    double sigma_P;
    double sigma_L;
    double sigma_D;
};

constexpr FreqInfo GPS_FREQ[3] = {
    {1575.42e6, CLIGHT / 1575.42e6, 0.3, 0.003, 0.5},   /* L1 */
    {1227.60e6, CLIGHT / 1227.60e6, 0.3, 0.003, 0.5},   /* L2 */
    {1176.45e6, CLIGHT / 1176.45e6, 0.2, 0.002, 0.4},   /* L5 */
};

constexpr double FREQ_L1   = GPS_FREQ[0].freq_hz;
constexpr double LAMBDA_L1 = GPS_FREQ[0].lambda;

constexpr double LAMBDA_WL = CLIGHT / (GPS_FREQ[0].freq_hz - GPS_FREQ[1].freq_hz);
constexpr double LAMBDA_NL = CLIGHT / (GPS_FREQ[0].freq_hz + GPS_FREQ[1].freq_hz);

constexpr double ALPHA_IF =  GPS_FREQ[0].freq_hz * GPS_FREQ[0].freq_hz
                           / (GPS_FREQ[0].freq_hz * GPS_FREQ[0].freq_hz
                            - GPS_FREQ[1].freq_hz * GPS_FREQ[1].freq_hz);
constexpr double BETA_IF  = -GPS_FREQ[1].freq_hz * GPS_FREQ[1].freq_hz
                           / (GPS_FREQ[0].freq_hz * GPS_FREQ[0].freq_hz
                            - GPS_FREQ[1].freq_hz * GPS_FREQ[1].freq_hz);

/* 参数块维数 ---------------------------------------------------------------------*/
constexpr int NUM_POSE      = 7;  /*< p(3) + q(4) */
constexpr int NUM_MIX       = 9;  /*< v(3) + bg(3) + ba(3) */
/* mix 参数块 = {v, bg, ba}。里程计刻度因子误差由 calib[0] 估计（见 OdoVelResidual），
 * 原先这里多放了一维 sodo，但没有任何残差读它 → 每帧一条恒零列、结构性秩亏，故退回 9 维。 */
constexpr int NUM_MIX_ODO   = 9;
constexpr int NUM_STATE     = 15; /*< 预积分残差维数 */
constexpr int NUM_STATE_ODO = 19; /*< 再加上里程位移(3) 与比例因子(1) */
constexpr int NUM_NOISE     = 12; /*< IMU 噪声维数 */
constexpr int NUM_NOISE_ODO = 16; /*< 再加上里程计白噪声(3) 与比例因子随机游走(1) */
constexpr int NUM_CALIB     = 14;  
/* calib[0..6]  = {sodo, abv_p, abv_y, lvOD_x, lvOD_y, lvOD_z, yaw_off}
 * calib[7..9]  = t_bc  (相机光心在 body 系下位置)
 * calib[10..13] = q_bc (qx, qy, qz, qw)  camera -> body */

static constexpr int STATE_DIM = NUM_STATE;  /*< 15 */
static constexpr int NOISE_DIM = NUM_NOISE;  /*< 12 */

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using CovMatrix   = Eigen::Matrix<double, STATE_DIM, STATE_DIM>;
using JacMatrix   = Eigen::Matrix<double, STATE_DIM, STATE_DIM>;
using NoiseMatrix = Eigen::Matrix<double, NOISE_DIM, NOISE_DIM>;
using GainMatrix  = Eigen::Matrix<double, STATE_DIM, NOISE_DIM>;
/* ============================================================
 * 15 维基础流形（p, q, v, bg, ba）
 * ============================================================ */
using DeltaN = Eigen::Matrix<double, NUM_STATE, 1>;   // 15×1
/* ============================================================
 * 16 维流形（在 15 维基础上扩展里程计比例因子 sodo）
 * ============================================================ */
using DeltaNOdo = Eigen::Matrix<double, NUM_STATE + 1, 1>;  // 16×1

inline bool isInfSentinel(double v) { return v > 2.0 * INF; }
template <class T>
inline void swapt(T &a, T &b) {
    T c = a;
    a   = b;
    b   = c;
}

enum AntMode {
    ANT_MODE_FB_B = 0,  /*< 基线前后、输出指车尾：原样 */
    ANT_MODE_FB_F = 1,  /*< 基线前后、输出指车头：再转 180° */
    ANT_MODE_LR_L = 2,  /*< 基线左右、输出指左侧：再转 +90° */
    ANT_MODE_LR_R = 3,  /*< 基线左右、输出指右侧：再转 -90° */
    ANT_MODE_ONE  = 4,  /*< 单天线，航向不可用 */
};

#define CC180C360(yaw)  ( (yaw)>0.0 ? (_2PI-(yaw)) : -(yaw) )   // counter-clockwise +-180deg -> clockwise 0~360deg for yaw
#define C360CC180(yaw)  ( (yaw)>=PI ? (_2PI-(yaw)) : -(yaw) )   // clockwise 0~360deg -> counter-clockwise +-180deg for yaw

class earth;
class Preintegration;

/* ---------- 常用常量（定义在 ipos_fixed.cpp） ---------- */
/* 常用常量：直接是 Eigen 对象（C++17 inline 变量） */
inline const Eigen::Vector3d    O31   = Eigen::Vector3d::Zero();
inline const Eigen::Vector3d    One31 = Eigen::Vector3d::Ones();
inline const Eigen::Matrix3d    I33   = Eigen::Matrix3d::Identity();
inline const Eigen::Quaterniond qI    = Eigen::Quaterniond::Identity();

Eigen::Matrix3d Rot(double angle, char axis);
Eigen::Matrix3d rcijk(const Eigen::Matrix3d &m, int ijk);
void normlize(Eigen::Quaterniond &q);

/* ============================================================================
 * 辅助：Vector3d/Quaterniond → Eigen 模板类型（支持 AutoDiff 的 Jet<T>）
 * ==========================================================================*/
template <typename T>
inline Eigen::Matrix<T, 3, 1> toEig3(const Eigen::Vector3d &v) {
    return Eigen::Matrix<T, 3, 1>(T(v.x()), T(v.y()), T(v.z()));
}
template <typename T>
inline Eigen::Quaternion<T> toEigQ(const Eigen::Quaterniond &q) {
    return Eigen::Quaternion<T>(T(q.w()), T(q.x()), T(q.y()), T(q.z()));
}

/* ============================ 模糊度键与管理 ============================
 *
 * 频率槽 slot：0..MAX_FREQ-1 为逐频（L1/L2/L5），100/200/300 为宽巷/窄巷/无电离层组合。
 * 弧段号 arc：同一 (卫星, 槽) 上的连续跟踪弧。发生周跳后 arc+1，
 *             新弧使用新的模糊度参数（旧弧的键保留到被清理为止）。
 *
 * 【为什么必须是 64 位】旧实现是
 *     ((ref_sat & 0x3F) << 16) | ((sat & 0x3F) << 8) | (freq & 0x0F)
 * 把卫星号截断到 6 bit(0..63)：任何编号相差 64 的卫星（多系统 sys*100+prn、
 * 北斗 C01–C63 等）都会**静默共用同一个模糊度参数**，而且键里没有系统号/弧段号。
 * 这里改成 16 bit 卫星号 + 8 bit 槽 + 24 bit 弧段。 */
enum AmbSlot {
    AMB_SLOT_L1 = 0,
    AMB_SLOT_L2 = 1,
    AMB_SLOT_L5 = 2,
    AMB_SLOT_WL = 100,
    AMB_SLOT_NL = 200,
    AMB_SLOT_IF = 300,
};

using AmbKey = uint64_t;
using AmbMap = std::map<AmbKey, std::array<double, 1>>;

inline AmbKey ambKey(int ref_sat, int sat, int slot, int arc = 0) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(ref_sat) & 0xFFFFu) << 48)
         | (static_cast<uint64_t>(static_cast<uint32_t>(sat)     & 0xFFFFu) << 32)
         | (static_cast<uint64_t>(static_cast<uint32_t>(slot)    & 0xFFu)   << 24)
         |  static_cast<uint64_t>(static_cast<uint32_t>(arc)     & 0xFFFFFFu);
}
inline int ambKeyRefSat(AmbKey k) { return static_cast<int>((k >> 48) & 0xFFFFu); }
inline int ambKeySat   (AmbKey k) { return static_cast<int>((k >> 32) & 0xFFFFu); }
inline int ambKeySlot  (AmbKey k) { return static_cast<int>((k >> 24) & 0xFFu); }
inline int ambKeyArc   (AmbKey k) { return static_cast<int>(k & 0xFFFFFFu); }

/* 参考星切换的整数转换：对每个卫星 i，
 *     N_{k'}^i = N_k^i − N_k^{k'}
 * 只有当 k' 相对旧参考 k 也存在模糊度时才能转换；否则新键以 0 起步（等价于重新收敛）。
 * 返回成功转换的卫星个数。 */
int ambPromoteReference(AmbMap &m, int old_ref, int new_ref, int slot, int arc = 0);

/* 周跳/断弧：把卫星 sat 在槽 slot 上的弧段推进到 new_arc。
 * 新键的初值取旧弧当前值（周跳后可用 slip_cycles 修正），返回新键。 */
AmbKey ambAdvanceArc(AmbMap &m, int ref_sat, int sat, int slot, int new_arc,
                     double slip_cycles = 0.0);

/* 清理本窗口未再出现的旧键（避免键集只增不减、影响协方差提取） */
void ambPruneUnused(AmbMap &m, const std::set<AmbKey> &used);

/* 把切空间 ∂r/∂δφ 投影成 ambient 四元数雅可比（PoseManifold::PlusJacobian
 * 是 0.5·Q(q)，两侧相乘后严格还原 ∂r/∂δφ） */
inline void projPhiToQAmbient(const Eigen::Vector3d& dr_dphi, const Eigen::Quaterniond& q, double* out4) {
    const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();
    out4[0] = 2.0 * ( dr_dphi.x() *  qw + dr_dphi.y() * (-qz) + dr_dphi.z() *  qy);
    out4[1] = 2.0 * ( dr_dphi.x() *  qz + dr_dphi.y() *  qw  + dr_dphi.z() * (-qx));
    out4[2] = 2.0 * ( dr_dphi.x() * (-qy) + dr_dphi.y() *  qx + dr_dphi.z() *  qw);
    out4[3] = 2.0 * ( dr_dphi.x() * (-qx) + dr_dphi.y() * (-qy) + dr_dphi.z() * (-qz));
}

/* ============================ 姿态 / 旋转 ============================ */
Eigen::Matrix3d a2mat(const Eigen::Vector3d &att);           /*< [pitch,roll,yaw]   -> Cnb  */
Eigen::Quaterniond a2qua(double pitch, double roll, double yaw);
Eigen::Quaterniond a2qua(const Eigen::Vector3d &att);
Eigen::Vector3d m2att(const Eigen::Matrix3d &Cnb);           /*< Cnb -> [pitch,roll,yaw]    */
Eigen::Vector3d q2att(const Eigen::Quaterniond &qnb);
void setYaw(Eigen::Quaterniond &q, double yaw = 0.0);

Eigen::Matrix3d ar2mat(const Eigen::Vector3d &attr);         /*< 反序欧拉角 -> Cnb          */
Eigen::Quaterniond ar2qua(const Eigen::Vector3d &attr);
Eigen::Vector3d m2attr(const Eigen::Matrix3d &Cnb);
Eigen::Vector3d q2attr(const Eigen::Quaterniond &qnb);

Eigen::Quaterniond rv2q(const Eigen::Vector3d &rv);             /*< 旋转矢量 -> 四元数         */
Eigen::Matrix3d rv2m(const Eigen::Vector3d &rv);
Eigen::Vector3d q2rv(const Eigen::Quaterniond &q);
Eigen::Vector3d m2rv(const Eigen::Matrix3d &Cnb);

/* 四元数右乘扰动的显式接口（对应旧 quat::operator+ / operator-）：
 *   q ⊕ φ = rv2q(-φ) ⊗ q ,  q ⊖ φ = rv2q(φ) ⊗ q
 * 单独写成自由函数而不是给 Eigen 类型重载运算符，避免污染 Eigen 的语义。 */
inline Eigen::Quaterniond qPlusPhi(const Eigen::Quaterniond &q, const Eigen::Vector3d &phi) {
    return rv2q(-phi) * q;
}

inline Eigen::Quaterniond qMinusPhi(const Eigen::Quaterniond &q, const Eigen::Vector3d &phi) {
    return rv2q(phi) * q;
}

Eigen::Matrix3d q2mat(const Eigen::Quaterniond &qnb);
Eigen::Quaterniond m2qua(const Eigen::Matrix3d &Cnb);
Eigen::Matrix3d askew(const Eigen::Vector3d &v);

Eigen::Vector3d qq2phi(const Eigen::Quaterniond &qcalcu, const Eigen::Quaterniond &qreal);
Eigen::Quaterniond addmu(const Eigen::Quaterniond &q, const Eigen::Vector3d &mu);
Eigen::Quaterniond UpDown(const Eigen::Quaterniond &q);
Eigen::Vector3d q2att(const Eigen::Quaterniond &qnb);

Eigen::Vector3d sv2att(const Eigen::Vector3d &fb, double yaw0, const Eigen::Vector3d &fn);
Eigen::Vector3d vn2att(const Eigen::Vector3d &vn);
double vn2att(double vel_east, double vel_north);
/* ============================ Matrix3d 线性代数 ============================ */
Eigen::Matrix3d Rot(double angle, char axis);
Eigen::Matrix3d rcijk(const Eigen::Matrix3d &m, int ijk);
double trMMT(const Eigen::Matrix3d &m1, const Eigen::Matrix3d &m2 = I33);
void symmetry(Eigen::Matrix3d &m);
Eigen::Matrix3d pow(const Eigen::Matrix3d &m, int k);
double trace(const Eigen::Matrix3d &m);
double det(const Eigen::Matrix3d &m);
Eigen::Matrix3d adj(const Eigen::Matrix3d &m);
Eigen::Matrix3d inv(const Eigen::Matrix3d &m);
Eigen::Vector3d diag(const Eigen::Matrix3d &m);
Eigen::Matrix3d diag(const Eigen::Vector3d &v);
Eigen::Matrix3d askew(const Eigen::Matrix3d &m, int I);
Eigen::Matrix3d dotmul(const Eigen::Matrix3d &m1, const Eigen::Matrix3d &m2);
Eigen::Matrix3d MMT(const Eigen::Matrix3d &m1, const Eigen::Matrix3d &m2);
Eigen::Matrix3d randn(const Eigen::Matrix3d &mu, const double &sigma);

/* ============================ Vector3d 标量 / 矢量工具 ============================ */
int IsZeros(const Eigen::Vector3d &v, double eps = EPS);
int IsZero(const double &val, double eps = EPS);
uint8_t IsZerosXY(const Eigen::Vector3d &v, double eps = EPS);
uint8_t IsNaN(const Eigen::Vector3d &v);
double diffYaw(double yaw, double yaw0);
Eigen::Vector3d abs(const Eigen::Vector3d &v);
Eigen::Vector3d maxabs(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);
double normXY(const Eigen::Vector3d &v);
double normXYInf(const Eigen::Vector3d &v);
Eigen::Vector3d sqrt(const Eigen::Vector3d &v);
Eigen::Vector3d pow(const Eigen::Vector3d &v, int k);
double dot(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);
Eigen::Vector3d dotmul(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);
Eigen::Matrix3d vxv(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);
double sinAng(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);
Eigen::Vector3d sort(const Eigen::Vector3d &v);
Eigen::Vector3d randn(const Eigen::Vector3d &mu, const Eigen::Vector3d &sigma);
double MKQt(double sR, double tau);
Eigen::Vector3d MKQt(const Eigen::Vector3d &sR, const Eigen::Vector3d &tau);

/* ============================ 动态向量线性代数（Eigen） ============================ */
double norm1(const double *pd, int n);
double normInf(const double *pd, int n);

Eigen::VectorXd absv(const Eigen::VectorXd &v);

/* norm / norm1 / normInf 用模板：这样 Vector3d、Matrix3d、VectorXd、MatrixXd
 * 以及各种 Eigen 表达式都能直接传，不会出现 "Vector3d 与 VectorXd 重载歧义"。 */
template <typename Derived>
inline double norm(const Eigen::MatrixBase<Derived> &v) { return v.norm(); }
template <typename Derived>
inline double norm1(const Eigen::MatrixBase<Derived> &v) { return v.template lpNorm<1>(); }
template <typename Derived>
inline double normInf(const Eigen::MatrixBase<Derived> &v) {
    return v.template lpNorm<Eigen::Infinity>();
}
Eigen::VectorXd pow(const Eigen::VectorXd &v, int k);
Eigen::VectorXd sort(const Eigen::VectorXd &v);
double          dot(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2);
Eigen::VectorXd dotmul(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2);
Eigen::VectorXd randn(const Eigen::VectorXd &mu, const Eigen::VectorXd &sigma);

/* ============================ 动态矩阵线性代数（Eigen） ============================ */
void            symmetry(Eigen::MatrixXd &m);
double          trace(const Eigen::MatrixXd &m);
Eigen::MatrixXd dotmul(const Eigen::MatrixXd &m1, const Eigen::MatrixXd &m2);
Eigen::VectorXd diag(const Eigen::MatrixXd &m);
Eigen::MatrixXd diag(const Eigen::VectorXd &v);
Eigen::MatrixXd eye(int n);
Eigen::MatrixXd inv4(const Eigen::MatrixXd &m);
Eigen::MatrixXd randn(const Eigen::MatrixXd &mu, const double &sigma);

/* 对称正定线性代数：输入 / 输出都是普通 Eigen 矩阵。
 * 返回 0×0 表示"不是正定 / 数值上过于病态"，调用方须自行走兜底路径。 */
Eigen::MatrixXd llt_L(const Eigen::MatrixXd &A);
Eigen::MatrixXd llt_sqrtinv(const Eigen::MatrixXd &A);
Eigen::MatrixXd llt_inv(const Eigen::MatrixXd &A);
Eigen::MatrixXd pinv_sym(const Eigen::MatrixXd &A, double eps = 1e-10);
void accumulate_normal_equations(const Eigen::MatrixXd &J, const Eigen::VectorXd &r,
                                 Eigen::MatrixXd &H, Eigen::VectorXd &b);

/* Schur 补 ----------------------------------------------------------------------
 *   H, b  : 完整法方程（前 num_marginalized 维是被边缘化的状态）
 *   J_out : n_keep × n_keep，满足 J_out^T J_out = H*
 *   r_out : n_keep，满足 J_out^T r_out = b*
 * 返回 false 表示维数非法或分解失败。 -------------------------------------------*/
bool schur_complement(const Eigen::MatrixXd &H, const Eigen::VectorXd &b, int num_marginalized,
                      Eigen::MatrixXd &J_out, Eigen::VectorXd &r_out);

/* ============================ 坐标系 ============================ */
Eigen::Matrix3d pos2Cen(const Eigen::Vector3d &pos);
Eigen::Vector3d xyz2blh(const Eigen::Vector3d &xyz);
Eigen::Vector3d blh2xyz(const Eigen::Vector3d &blh);
Eigen::Vector3d Vxyz2enu(const Eigen::Vector3d &Vxyz, const Eigen::Vector3d &pos);


class earth {
public:
    double a;      /*< 长半轴 */
    double b;      /*< 短半轴 */
    double f;      /*< 扁率 */
    double wie;    /*< 地球自转角速率 */
    double sl;     /*< sin(lat) */
    double sl2;    /*< sin^2(lat) */
    double sl4;    /*< sin^4(lat) */
    double cl;     /*< cos(lat) */
    double tl;     /*< tan(lat) */
    double RMh;    /*< 子午圈曲率半径 + 高程 */
    double RNh;    /*< 卯酉圈曲率半径 + 高程 */
    double clRNh;  /*< cos(lat) * RNh */
    double f_RMh;  /*< 1 / RMh */
    double f_RNh;  /*< 1 / RNh */
    double f_clRNh; /*< 1 / clRNh */

    Eigen::Vector3d pos;  /*< {lat, lon, h} */
    Eigen::Vector3d vn;   /*< {vE, vN, vU} */
    Eigen::Vector3d wnie; /*< 地球自转在 n 系下的表示 */
    Eigen::Vector3d wnen; /*< n 系相对 e 系的转动（运输率） */
    Eigen::Vector3d wnin; /*< n 系相对 i 系的转动 */
    Eigen::Vector3d gn;   /*< 重力矢量（ENU: {0,0,-g}） */
    Eigen::Vector3d gcc;  /*< 重力 + 科氏/向心修正 */
    Eigen::Vector3d *pgn; /*< 若不为空则用它替换 gn */

    earth(double a = RE, double f = f0_earth);

    void Init(double a = RE, double f = f0_earth);

    /* 由位置和速度刷新椭球/重力/角速度参数 -----------------------------------------*/
    void Update(const Eigen::Vector3d &pos, const Eigen::Vector3d &vn);

    /* 速度 -> 位置增量（{dLat, dLon, dh}） -----------------------------------------*/
    Eigen::Vector3d vn2dpos(const Eigen::Vector3d &vn, float ts = 1.0f) const;
};

class LocalFrame {
public:
    LocalFrame(void) { init(Eigen::Vector3d::Zero()); }

    /* 以 blh0 为原点建立局部 NEU 系 ----------------------------------------------*/
    void init(const Eigen::Vector3d &blh0);

    bool isInit(void) const { return init_; }
    const Eigen::Vector3d &blh0(void) const { return blh0_; }

    Eigen::Vector3d blh2neu(const Eigen::Vector3d &blh) const;
    Eigen::Vector3d neu2blh(const Eigen::Vector3d &neu) const;

    /* 参考点处的 earth 模型（wnie / gn / RMh / RNh 都取这里的值） ------------------*/
    const earth &earthModel(void) const { return eth_; }

    /* ENU 下的重力矢量 {0, 0, -g} ------------------------------------------------*/
    const Eigen::Vector3d &gravity(void) const { return gravity_; }

    /* ENU 下的地球自转 {0, wie*cos(lat0), wie*sin(lat0)} ------------------------*/
    const Eigen::Vector3d &wnie(void) const { return eth_.wnie; }

    const Eigen::Matrix3d&  R_en_m3(void)   const { return R_en_m3_; }
    const Eigen::Vector3d& p0Ecef_v3(void) const { return p0_ecef_v3_; }

private:
    bool init_{false};
    Eigen::Vector3d blh0_;
    Eigen::Vector3d gravity_;
    earth eth_;

    Eigen::Matrix3d  R_en_m3_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p0_ecef_v3_ = Eigen::Vector3d::Zero();
};


class IMU {
public:
    double tk;       /*< 当前时间 */
    float nts;       /*< 本次更新覆盖的时间 */
    float _nts;      /*< 1 / nts */
    Eigen::Matrix3d Kg;         /*< 陀螺标定阵 */
    Eigen::Vector3d eb;        /*< 陀螺零偏 */
    Eigen::Matrix3d Ka;         /*< 加表标定阵 */
    Eigen::Vector3d db;        /*< 加表零偏 */

    Eigen::Vector3d phim;      /*< 角度增量（已含圆锥补偿） */
    Eigen::Vector3d dvbm;      /*< 速度增量（已含划桨补偿） */
    Eigen::Vector3d wmm;       /*< 角速率测量之和 */
    Eigen::Vector3d vmm;       /*< 比力测量之和 */
    Eigen::Vector3d swmm;      /*< 角速率累加 */
    Eigen::Vector3d svmm;      /*< 比力累加 */
    Eigen::Vector3d wm_1;      /*< 上一采样角增量 */
    Eigen::Vector3d vm_1;      /*< 上一采样速度增量 */

    int nSamples;    /*< 本次更新的子样数（1~5） */
    bool preFirst;   /*< 是否为上电后的第一次更新 */
    bool onePlusPre; /*< 单子样时是否补上上一采样 */
    bool preWb;      /*< 是否已有上一采样 */

    IMU(void);

    /* 复位（保留标定参数），onePlusPre 默认打开 --------------------------------*/
    void Reset(void);

    void SetKga(const Eigen::Matrix3d &Kg0, const Eigen::Vector3d eb0, const Eigen::Matrix3d &Ka0, const Eigen::Vector3d &db0);

    /* 由等间隔采样增量计算 phim / dvbm
     *   pwm[i] : 第 i 个子样的角增量  (rad)
     *   pvm[i] : 第 i 个子样的速度增量 (m/s)
     *   nSamples: 1~5，ts: 单个子样的时间
     * 返回更新后的时间 tk -----------------------------------------------------------*/
    double Update(const Eigen::Vector3d *pwm, const Eigen::Vector3d *pvm, int nSamples, double ts);
};

struct PreintegrationParam {
    double gyr_arw{0.0};      /*< 角度随机游走 rad/sqrt(s) */
    double acc_vrw{0.0};      /*< 速度随机游走 m/s^1.5 */
    double gyr_bias_std{0.0}; /*< 陀螺零偏标准差 rad/s */
    double acc_bias_std{0.0}; /*< 加表零偏标准差 m/s^2 */
    double corr_time{3600.0}; /*< 零偏一阶马尔可夫相关时间 s */

    /* ---- 里程计（use_odometer = true 时生效）---- */
    Eigen::Vector3d odo_std;             /*< 单次里程增量的白噪声标准差 (m) */
    double odo_srw{0.0};       /*< 里程计比例因子随机游走 (1/sqrt(s)) */
    Eigen::Vector3d abv;                 /*< 安装角 [pitch, 0, yaw]，与 ipos3g 的 odo::ODKappa 一致 */
    Eigen::Vector3d lvOD;                /*< 里程计杆臂（b 系，m） */
};

/* 关键帧状态 ---------------------------------------------------------------------*/
struct State {
    double time{0.0};
    Eigen::Vector3d p;                       /*< 局部 NEU 位置 (m) */
    Eigen::Quaterniond q{1.0, 0.0, 0.0, 0.0};    /*< b -> n */
    Eigen::Vector3d v;                       /*< n 系速度 (m/s) */
    Eigen::Vector3d bg;                      /*< 陀螺零偏 (rad/s) */
    Eigen::Vector3d ba;                      /*< 加表零偏 (m/s^2) */
    double sodo{0.0};              /*< 里程计比例因子误差（无量纲） */
    Eigen::Vector3d s;                       /*< 预积分里程位移（b0 系，m），仅内部使用 */
};

/* 观测 ---------------------------------------------------------------------------*/
struct ImuMeas {
    double time{0.0};
    double dt{0.0};
    Eigen::Vector3d dtheta; /*< 角增量 (rad) */
    Eigen::Vector3d dvel;   /*< 速度增量 (m/s) */
    double odovel{0.0}; /*< 本次采样间隔内的里程增量 (m)，对应 OB_GINS 的 imu.odovel */
};

struct GnssMeas {
    double time{0.0};
    Eigen::Vector3d blh;    /*< {lat, lon, h}，弧度/米 */
    Eigen::Vector3d std;    /*< 位置 {sigma_E, sigma_N, sigma_U}，米（局部 NEU） */
    Eigen::Vector3d vn;     /*< n 系速度 (m/s)，需要时填（ipos3g 的 KF 就用它） */
    Eigen::Vector3d vn_std; /*< 速度标准差 (m/s) */
    double yaw{0.0};     /*< 双天线航向 (rad)，yaw_std<=0 表示无效 */
    double yaw_std{0.0}; /*< 航向标准差 (rad) */
};

struct OdoMeas {
    double time{0.0};
    double ds{0.0}; /*< 里程增量 (m) */
    double dt{0.0};
};

/* ---- 归一化平面上的视觉观测（后端因子使用） ---- */
struct VisualObs {
    int    feat_id{0};
    Eigen::Vector3d  xyz_c{0.0, 0.0, 0.0};   /* (X/Z, Y/Z, 1) */
    double sigma{1.5 / 460.0};
};

struct SatState {
    int    sat_id{0};
    Eigen::Vector3d  pos_ecef;
    Eigen::Vector3d  vel_ecef;
    double clk_bias_s{0.0};
    double clk_drift_sps{0.0};
    double elev_rad{0.0};
    bool   valid{false};
};

struct DDObs {
    int    ref_sat{0};
    int    sat{0};
    double dd_P[MAX_FREQ]{};
    double dd_L[MAX_FREQ]{};
    double dd_D[MAX_FREQ]{};
    bool   has_P[MAX_FREQ]{};
    bool   has_L[MAX_FREQ]{};
    bool   has_D[MAX_FREQ]{};
    double sigma_P{0.6};
    double sigma_L{0.01};
    SatState sat_i, sat_k;
};

/* ---- 相机针孔模型 + 径向切向畸变 ---- */
struct CameraModel {
    double fx{0.0}, fy{0.0}, cx{0.0}, cy{0.0};
    double k1{0.0}, k2{0.0}, p1{0.0}, p2{0.0};
    cv::Mat K() const {
        return cv::Mat(cv::Matx33d(fx, 0,  cx,
                                0,  fy, cy,
                                0,  0,  1));
    }

    cv::Mat D() const {
        return cv::Mat(cv::Matx14d(k1, k2, p1, p2));
    }
};

/* ---- 单个被追踪特征 ---- */
struct TrackedFeature {
    int             feat_id{0};
    cv::Point2f     pt{};
    Eigen::Vector3d xyz_c{0,0,1};
    int             track_cnt{1};
};

/* ---- 视觉前端（Shi-Tomasi + LK + RANSAC F）---- */
class VisualFrontend {
public:
    VisualFrontend(const CameraModel& cam, int max_feat = 150, int min_dist = 20);
    std::vector<TrackedFeature> process(double t, const cv::Mat& gray);
    const CameraModel& camera() const { return cam_; }

private:
    cv::Mat makeMask(const std::vector<cv::Point2f>& pts) const;
    void    rejectWithF(const std::vector<cv::Point2f>& p0,
                        const std::vector<cv::Point2f>& p1,
                        std::vector<uchar>& status);
    Eigen::Vector3d undistortToNorm(const cv::Point2f& pt) const;

    CameraModel cam_;
    int         max_feat_, min_dist_;
    cv::Mat                  prev_img_;
    std::vector<cv::Point2f> prev_pts_;
    std::vector<int>         prev_ids_, track_cnt_;
    int                      next_id_{1};
    bool                     first_frame_{true};
};

/* ============================================================================
 * 全局标定状态（滑窗内所有帧共享同一份，不随时间变化）
 *
 * 参数：sodo, abv_pitch, abv_yaw, lvOD(3), yaw_gnss_offset
 * 参数化：全部加性，无需流形
 * 数据布局：calib[0..6] = {sodo, ap, ay, lx, ly, lz, yoff}
 * ==========================================================================*/
struct CalibState {
    double sodo{0.0};
    double abv_pitch{0.0};
    double abv_yaw{0.0};
    Eigen::Vector3d  lvOD{0.0, 0.0, 0.0};
    double yaw_gnss_offset{0.0};
    /* ---- 相机外参（新增） ---- */
    Eigen::Vector3d  t_bc{0.0, 0.0, 0.0};
    Eigen::Quaterniond   q_bc{1.0, 0.0, 0.0, 0.0};

    void toData(double *d) const {
        d[0]=sodo; d[1]=abv_pitch; d[2]=abv_yaw;
        d[3]=lvOD.x(); d[4]=lvOD.y(); d[5]=lvOD.z();
        d[6]=yaw_gnss_offset;
        d[7]=t_bc.x(); d[8]=t_bc.y(); d[9]=t_bc.z();
        d[10]=q_bc.x(); d[11]=q_bc.y(); d[12]=q_bc.z(); d[13]=q_bc.w();
    }
    void fromData(const double *d) {
        sodo=d[0]; abv_pitch=d[1]; abv_yaw=d[2];
        lvOD=Eigen::Vector3d(d[3],d[4],d[5]); yaw_gnss_offset=d[6];
        t_bc=Eigen::Vector3d(d[7],d[8],d[9]);
        q_bc=Eigen::Quaterniond(d[13],d[10],d[11],d[12]);
    }
};

/* 一个关键帧 ----------------------------------------------------------------------*/
struct Frame {
    double time{0.0};
    /* Ceres 参数块（顺序与 gopt_types.h 一致）。数组固定长度，保证地址稳定 ----*/
    std::array<double, NUM_POSE> pose{};     /*< {pE,pN,pU, qx,qy,qz,qw} */
    std::array<double, NUM_MIX_ODO> mix{};   /*< {v, bg, ba} */

    /* 到下一关键帧的预积分（最后一帧为空） ----------------------------------*/
    std::shared_ptr<Preintegration> pre_to_next;

    /* 该帧上的 GNSS 位置观测（已转到局部 ENU） -------------------------------*/
    bool has_gnss{false};
    Eigen::Vector3d gnss_pos{};
    Eigen::Vector3d gnss_std{1.0, 1.0, 1.0};
    Eigen::Vector3d gnss_std0{1.0, 1.0, 1.0}; /*< 原始 std，重加权时以它为基准（避免累乘） */
    double gnss_time{0.0};
    bool   gnss_in_gap{false};      /*< 该帧落在模拟的 GNSS 中断区间（只做参考，不入因子） */

    /* 该帧上的 GNSS 速度观测 ------------------------------------*/
    bool has_gnss_vel{false};
    Eigen::Vector3d gnss_vn{};
    Eigen::Vector3d gnss_vn_std{1.0, 1.0, 1.0};

    /* 静止段标记（由 IMU 判据得到，见 Estimator::updateStaticDetector） -------------*/
    bool is_static{false};
    bool has_yaw_hold{false};
    double yaw_ref{0.0};

    /* 双天线航向观测 ------------------------------------------------------------*/
    bool has_gnss_yaw{false};
    double gnss_yaw{0.0};
    double gnss_yaw_std{0.0};

    /* 原始双天线航向（已按天线安装方式折算到内部 yaw）。无论是否启用航向因子
     * 都会记录，用于统计"估计航向 vs 双天线航向"的一致性。 */
    bool   has_raw_yaw{false};
    double raw_yaw_conv{0.0};

    /* FEJ：该帧的线性化点 --------------------*/
    bool has_lin{false};
    std::array<double, NUM_POSE> pose_lin{};
    std::array<double, NUM_MIX_ODO> mix_lin{};

    /* 里程计速度观测（前向速度，b 系） ------------------------------------ */
    bool   has_odo{false};
    double odo_dS{0.0};               /*< 里程增量 (m) */
    double odo_dt{0.0};               /*< 对应时间间隔 (s) */
    Eigen::Vector3d  odo_omega_meas{};          /*< 该段陀螺原始测量 (rad/s)，用于杆臂补偿 */
    double odo_std{0.05};             /*< 里程速度标准差 (m/s) */

    std::vector<VisualObs> visual_obs;

    /* ---- GNSS 紧组合 ---- */
    bool has_gnss_raw{false};
    std::vector<DDObs>  dd_obs;
    /* 与 dd_obs 并行：每 (DD, slot) 对应的弧段号。周跳时该 (ref_sat,sat,slot)
     * 的弧段号 +1，用于给模糊度参数生成稳定的键；不这样做的话一旦周跳
     * 就会复用旧弧的模糊度参数，整周解会被污染。 */
    std::vector<std::array<int, MAX_FREQ>> dd_arc;
    std::vector<std::array<double, MAX_FREQ>> amb_values;
    std::vector<std::array<bool,   MAX_FREQ>> amb_fixed;
    std::vector<double> amb_wl_values;
    std::vector<bool>   amb_wl_fixed;
    std::vector<double> amb_if_values;
    std::vector<bool>   amb_if_fixed;
    std::array<double, 1> clk_bias{{0.0}};
    std::array<double, 1> clk_drift{{0.0}};
};

/* ============================================================================
 * PoseManifold：pose = [p(3), qx,qy,qz,qw]，右扰动，切空间 6 维
 * ==========================================================================*/
class PoseManifold : public ceres::Manifold {
public:
    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }

    bool Plus(const double *x, const double *delta, double *x_plus) const override {
        x_plus[0] = x[0] + delta[0];
        x_plus[1] = x[1] + delta[1];
        x_plus[2] = x[2] + delta[2];

        Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
        Eigen::Quaterniond dq = rv2q(Eigen::Vector3d(delta[3], delta[4], delta[5]));
        Eigen::Quaterniond q_new = q * dq;
        normlize(q_new);

        x_plus[3] = q_new.x();
        x_plus[4] = q_new.y();
        x_plus[5] = q_new.z();
        x_plus[6] = q_new.w();
        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

        const double qx = x[3], qy = x[4], qz = x[5], qw = x[6];
        /* ∂q/∂δφ = 0.5 · [ -q_v  q_w I + [q_v]× ]  转置到 [x,y,z,w] 顺序 */
        Eigen::Matrix<double, 4, 3> dq_dphi;
        dq_dphi << -qx, -qy, -qz,
                    qw, -qz,  qy,
                    qz,  qw, -qx,
                   -qy,  qx,  qw;
        dq_dphi *= 0.5;

        /* dq_dphi 行序 [w, x, y, z]，pose 的 [x, y, z, w] */
        J.block<1, 3>(3, 3) = dq_dphi.row(1);   // pose[3] = qx
        J.block<1, 3>(4, 3) = dq_dphi.row(2);   // pose[4] = qy
        J.block<1, 3>(5, 3) = dq_dphi.row(3);   // pose[5] = qz
        J.block<1, 3>(6, 3) = dq_dphi.row(0);   // pose[6] = qw
        return true;
    }
    bool Minus(const double *y, const double *x, double *y_minus_x) const override {
        y_minus_x[0] = y[0] - x[0];
        y_minus_x[1] = y[1] - x[1];
        y_minus_x[2] = y[2] - x[2];

        Eigen::Quaterniond qx(x[6], x[3], x[4], x[5]);
        Eigen::Quaterniond qy(y[6], y[3], y[4], y[5]);
        Eigen::Vector3d dphi = q2rv(((qx.conjugate())) * qy);
        y_minus_x[3] = dphi.x();
        y_minus_x[4] = dphi.y();
        y_minus_x[5] = dphi.z();
        return true;
    }

    /* ---- 新增：MinusJacobian ---- */
    bool MinusJacobian(const double *x, double *jacobian) const override {
        /* 形状 6×7（行主序）
         * ∂(y ⊖ x)/∂y|_{y=x}
         *   = [ I_3          0_{3×4} ]
         *     [ 0_{3×3}      J_q     ]
         * 其中 J_q 与 Ceres 内置 QuaternionManifold 保持一致 */
        Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        /* 位置部分 */
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

        /* 旋转部分：pose 中四元数顺序为 [qx, qy, qz, qw]
         *
         * Minus(y,x) = Log(x⁻¹⊗y)，令 y = x + dx，则
         *   vec(x⁻¹⊗dx) = w·dv − dw·v − v×dv
         *   δφ = 2·(w·dv − dw·v − v×dv)
         * 于是 ∂δφ/∂(qx,qy,qz,qw) = 2·Qᵀ = 下面这一份。
         * 必须与 CalibManifold::MinusJacobian 同式，且满足
         * MinusJacobian·PlusJacobian = I_6。 */
        const double qx = x[3], qy = x[4], qz = x[5], qw = x[6];

        Eigen::Matrix<double, 3, 4> Jq;
        Jq <<  qw,  qz, -qy, -qx,
              -qz,  qw,  qx, -qy,
               qy, -qx,  qw, -qz;
        Jq *= 2.0;

        J.block<3, 4>(3, 3) = Jq;
        return true;
    }
};

/* ============================================================================
 * CalibManifold：NUM_CALIB=14 维参数块的流形
 *   前 10 维（sodo, ap, ay, lvx, lvy, lvz, yoff, tbx, tby, tbz）加性
 *   后 4 维（qbx, qby, qbz, qbw）用四元数流形（切空间 3 维）
 *   总 ambient = 14, tangent = 13
 * ==========================================================================*/
class CalibManifold : public ceres::Manifold {
public:
    int AmbientSize() const override { return NUM_CALIB; }
    int TangentSize() const override { return NUM_CALIB - 1; }

    bool Plus(const double* x, const double* delta, double* x_plus) const override {
        for (int i = 0; i < 10; ++i) x_plus[i] = x[i] + delta[i];
        Eigen::Quaterniond q(x[13], x[10], x[11], x[12]);   /* (w, x, y, z) */
        Eigen::Quaterniond dq = rv2q(Eigen::Vector3d(delta[10], delta[11], delta[12]));
        Eigen::Quaterniond q_new = q * dq;
        normlize(q_new);
        x_plus[10] = q_new.x();
        x_plus[11] = q_new.y();
        x_plus[12] = q_new.z();
        x_plus[13] = q_new.w();
        return true;
    }

    bool PlusJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
                                 Eigen::Dynamic, Eigen::RowMajor>>
            J(jacobian, NUM_CALIB, NUM_CALIB - 1);
        J.setZero();
        for (int i = 0; i < 10; ++i) J(i, i) = 1.0;
        const double qx = x[10], qy = x[11], qz = x[12], qw = x[13];
        /* 0.5 * Q(q_bc)，行序 [qx, qy, qz, qw] */
        J(10,10) =  0.5*qw; J(10,11) = -0.5*qz; J(10,12) =  0.5*qy;
        J(11,10) =  0.5*qz; J(11,11) =  0.5*qw; J(11,12) = -0.5*qx;
        J(12,10) = -0.5*qy; J(12,11) =  0.5*qx; J(12,12) =  0.5*qw;
        J(13,10) = -0.5*qx; J(13,11) = -0.5*qy; J(13,12) = -0.5*qz;
        return true;
    }

    bool Minus(const double* y, const double* x, double* y_minus_x) const override {
        for (int i = 0; i < 10; ++i) y_minus_x[i] = y[i] - x[i];
        Eigen::Quaterniond qy(y[13], y[10], y[11], y[12]);
        Eigen::Quaterniond qx(x[13], x[10], x[11], x[12]);
        Eigen::Vector3d dphi = q2rv(((qx.conjugate())) * qy);
        y_minus_x[10] = dphi.x();
        y_minus_x[11] = dphi.y();
        y_minus_x[12] = dphi.z();
        return true;
    }

    bool MinusJacobian(const double* x, double* jacobian) const override {
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::RowMajor>>
            J(jacobian, NUM_CALIB - 1, NUM_CALIB);
        J.setZero();
        for (int i = 0; i < 10; ++i) J(i, i) = 1.0;

        const double qx = x[10], qy = x[11], qz = x[12], qw = x[13];

        /* ∂δφ/∂q = 2·Q(q)ᵀ，列序 [qx, qy, qz, qw]：
        *   δφx: ( 2qw,  2qz, -2qy, -2qx)
        *   δφy: (-2qz,  2qw,  2qx, -2qy)
        *   δφz: ( 2qy, -2qx,  2qw, -2qz) */
        J(10,10) =  2*qw; J(10,11) =  2*qz; J(10,12) = -2*qy; J(10,13) = -2*qx;
        J(11,10) = -2*qz; J(11,11) =  2*qw; J(11,12) =  2*qx; J(11,13) = -2*qy;
        J(12,10) =  2*qy; J(12,11) = -2*qx; J(12,12) =  2*qw; J(12,13) = -2*qz;
        return true;
    }
};

/* ============================================================================
 * Preintegration
 * ----------------------------------------------------------------------------
 * 预积分量都在起始关键帧的 b0 系中：
 *     Δp_ij, Δv_ij (m, m/s), Δq_ij (b0 -> b_i)
 * 状态误差顺序（与 NUM_STATE = 15 对应）：
 *     δx = [δp, δv, δφ, δbg, δba]
 * 噪声向量顺序（与 NUM_NOISE = 12 对应）：
 *     w  = [ng, na, nbg, nba]
 * ==========================================================================*/
class Preintegration {
public:
    Preintegration(void);
    explicit Preintegration(const PreintegrationParam &param);

    /* ---- 生命周期 ---- */
    void reset(void);                                     /*< 清空所有状态与测量 */
    void setParam(const PreintegrationParam &param) { param_ = param; }

    /* ---- 增量式预积分（外部每来一帧 IMU 调用一次） ---- */
    void integration(const ImuMeas &meas);

    /* ---- 用当前 bg_/ba_ 对已缓存测量重放（bias 更新后调用） ---- */
    void repropagation(void);

    /* ---- 设置线性化点处的零偏（会触发 repropagation） ---- */
    void setBias(const Eigen::Vector3d &bg, const Eigen::Vector3d &ba);

    /* ---- 丢弃 repropagation 用的历史测量 ----
     * 滑窗优化里 pre_to_next 只用来求残差，不会再改零偏重放，
     * 因此冻结时可以丢掉这段历史，避免每帧拷贝 ~200 条 ImuMeas。 */
    void dropHistory(void) { meas_list_.clear(); }

    /* ---- 访问器 ---- */
    double        dt(void) const { return dt_; }
    int           num(void) const { return num_; }
    const Eigen::Vector3d  &p(void) const { return p_; }   /*< Δp_ij */
    const Eigen::Vector3d  &v(void) const { return v_; }   /*< Δv_ij */
    const Eigen::Quaterniond   &q(void) const { return q_; }   /*< Δq_ij */
    const Eigen::Vector3d  &bg(void) const { return bg_; }
    const Eigen::Vector3d  &ba(void) const { return ba_; }
    const Eigen::Vector3d &s(void)    const { return s_; }
    double       sodo(void) const { return sodo_; }
    const CovMatrix &cov(void) const { return cov_; }
    const JacMatrix &jac(void) const { return jac_; }

    /* 残余残差项的快捷访问（供图优化残差使用） */
    Eigen::Matrix3d R0(void) const { return qI.toRotationMatrix(); }

private:
    void propagate(const ImuMeas &meas);
    /* 构造 F、G、Q */
    CovMatrix   buildF(const Eigen::Vector3d &a_body,
                       const Eigen::Vector3d &w_body,
                       const Eigen::Matrix3d &R,
                       double dt) const;
    GainMatrix  buildG(const Eigen::Matrix3d &R) const;
    NoiseMatrix buildQ(double dt) const;

    Eigen::Vector3d  s_{O31};      /*< 预积分里程位移（b0 系） */
    double sodo_{0.0};   /*< 里程比例因子误差 */

    PreintegrationParam param_;
    std::deque<ImuMeas> meas_list_;  /*< 用于 repropagation 的历史测量 */

    /* ---- 累加状态 ---- */
    double dt_   = 0.0;
    int    num_  = 0;
    Eigen::Vector3d  p_{O31};
    Eigen::Vector3d  v_{O31};
    Eigen::Quaterniond   q_{qI};

    /* ---- 线性化点处的零偏 ---- */
    Eigen::Vector3d  bg_{O31};
    Eigen::Vector3d  ba_{O31};

    /* ---- 中值积分缓存 ---- */
    Eigen::Vector3d  last_dtheta_{O31};
    Eigen::Vector3d  last_dvel_{O31};
    double last_dt_ = 0.0;
    bool   first_   = true;

    /* ---- 协方差 / 雅可比 ---- */
    CovMatrix cov_ = CovMatrix::Zero();
    JacMatrix jac_ = JacMatrix::Identity();
};

/* ============================================================
 * 高层封装：管理若干残差块，按帧粒度边缘化
 * ============================================================ */
class MarginalizationInfo {
public:
    MarginalizationInfo() = default;

    /* 清空累积状态 */
    void reset();

    /* 追加一个残差块：H += Jᵀ J, b += Jᵀ r */
    void addResidual(const Eigen::MatrixXd &J, const Eigen::VectorXd &r);

    /* 边缘化前 num_marginalized 维。成功返回 true。
     * 若之前已 call 过 setPrior，会在原有先验基础上继续累积。 */
    bool marginalize(int num_marginalized);

    /* 把外部的先验（例如上一次边缘化的输出）作为新的残差加入。
     *   通常用于跨窗口传递：把 (J_out, r_out) 作为第一个残差块。 */
    void setPrior(const Eigen::MatrixXd &J_out, const Eigen::VectorXd &r_out);

    /* 访问器 */
    int  dim(void) const { return n_; }
    const Eigen::MatrixXd  &H(void) const { return H_; }
    const Eigen::VectorXd  &b(void) const { return b_; }
    const Eigen::MatrixXd  &J_prior(void) const { return J_prior_; }
    const Eigen::VectorXd  &r_prior(void) const { return r_prior_; }

private:
    int  n_{0};
    Eigen::MatrixXd  H_{};
    Eigen::VectorXd  b_{};
    /* 上一次边缘化得到的先验 */
    Eigen::MatrixXd  J_prior_{};
    Eigen::VectorXd  r_prior_{};
};


/* ============================================================================
 * 因子 1：IMU 预积分（15 维）
 *
 * 残差向量（与 NUM_STATE = 15 对应）：
 *     r = [ δΔp ; δΔv ; δΔφ ; bg_j-bg_i ; ba_j-ba_i ]
 * 其中 Δp/Δv/Δq 先按"当前零偏 - 线性化零偏"做一阶补偿：
 *     Δp ← Δp + ∂Δp/∂ba·δba + ∂Δp/∂bg·δbg
 *     Δv ← Δv + ∂Δv/∂ba·δba + ∂Δv/∂bg·δbg
 *     Δq ← Δq ⊗ Exp(∂δφ/∂bg·δbg)
 * 偏导直接取预积分累积的状态转移矩阵 Φ 的对应分块（Φ = ∂δx_j/∂δx_i），
 * 与 OB_GINS 的 PreintegrationFactor 一致；没有这一步时零偏在优化里
 * 是完全不可观的（残差对 bg/ba 的导数为 0）。
 *
 * whiten = true 时再乘 S = sqrt_information，S·P·Sᵀ = I（P 为预积分协方差）。
 * ==========================================================================*/
struct PreintResidual {
    PreintResidual(const Preintegration &p, const Eigen::Vector3d &g, bool whiten, bool use_bias_jac)
        : preint_(p), g_n_(g) {
        /* 线性化点处的零偏 */
        bg_lin_ = p.bg();
        ba_lin_ = p.ba();

        /* 零偏雅可比：Φ 的分块 */
        const Eigen::Matrix<double, 15, 15> &J = p.jac();
        Jp_bg_ = J.block<3, 3>(0, 9);
        Jp_ba_ = J.block<3, 3>(0, 12);
        Jv_bg_ = J.block<3, 3>(3, 9);
        Jv_ba_ = J.block<3, 3>(3, 12);
        Jq_bg_ = J.block<3, 3>(6, 9);

        /* 不使用零偏雅可比时 J 全零，残差等价于旧版（零偏不可观） */
        if (!use_bias_jac) {
            Jp_bg_.setZero(); Jp_ba_.setZero();
            Jv_bg_.setZero(); Jv_ba_.setZero();
            Jq_bg_.setZero();
        }

        if (whiten) {
            const Eigen::MatrixXd S = llt_sqrtinv(p.cov());
            if (S.rows() == 15) {
                S_ = S;
                whiten_ = true;
            }
        }
    }

    template <typename T>
    bool operator()(const T *const pose_i, const T *const mix_i,
                    const T *const pose_j, const T *const mix_j,
                    T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_i(pose_i);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_j(pose_j);
        Eigen::Quaternion<T> q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Quaternion<T> q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_i (mix_i);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_j (mix_j);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_i(mix_i + 3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_j(mix_j + 3);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_i(mix_i + 6);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_j(mix_j + 6);

        const T dt = T(preint_.dt());
        const Eigen::Matrix<T, 3, 1> g_n = toEig3<T>(g_n_);

        /* ---- 零偏一阶补偿 ---- */
        const Eigen::Matrix<T, 3, 1> dbg = bg_i - bg_lin_.template cast<T>();
        const Eigen::Matrix<T, 3, 1> dba = ba_i - ba_lin_.template cast<T>();

        Eigen::Matrix<T, 3, 1> dp = toEig3<T>(preint_.p());
        Eigen::Matrix<T, 3, 1> dv = toEig3<T>(preint_.v());
        Eigen::Quaternion<T>   dq = toEigQ<T>(preint_.q());

        dp += Jp_bg_.template cast<T>() * dbg + Jp_ba_.template cast<T>() * dba;
        dv += Jv_bg_.template cast<T>() * dbg + Jv_ba_.template cast<T>() * dba;

        const Eigen::Matrix<T, 3, 1> dphi = Jq_bg_.template cast<T>() * dbg;
        const Eigen::Quaternion<T> dq_corr(T(1.0), T(0.5) * dphi(0),
                                           T(0.5) * dphi(1), T(0.5) * dphi(2));
        dq = (dq * dq_corr).normalized();

        const Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();

        Eigen::Map<Eigen::Matrix<T, 15, 1>> r(residual);

        r.template segment<3>(0) = R_i.transpose() *
            (p_j - p_i - v_i * dt - T(0.5) * g_n * dt * dt) - dp;
        r.template segment<3>(3) = R_i.transpose() *
            (v_j - v_i - g_n * dt) - dv;

        const Eigen::Quaternion<T> q_rel = q_i.conjugate() * q_j;
        const Eigen::Quaternion<T> q_err = dq.conjugate() * q_rel;
        r.template segment<3>(6) = T(2.0) * q_err.vec();

        r.template segment<3>(9)  = bg_j - bg_i;
        r.template segment<3>(12) = ba_j - ba_i;

        if (whiten_) {
            r = S_.template cast<T>() * r;
        }
        return true;
    }

    Preintegration preint_;
    Eigen::Vector3d         g_n_;
    Eigen::Matrix<double, 3, 3> Jp_bg_{Eigen::Matrix3d::Zero()}, Jp_ba_{Eigen::Matrix3d::Zero()};
    Eigen::Matrix<double, 3, 3> Jv_bg_{Eigen::Matrix3d::Zero()}, Jv_ba_{Eigen::Matrix3d::Zero()};
    Eigen::Matrix<double, 3, 3> Jq_bg_{Eigen::Matrix3d::Zero()};
    Eigen::Vector3d bg_lin_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d ba_lin_{Eigen::Vector3d::Zero()};
    Eigen::Matrix<double, 15, 15> S_{Eigen::Matrix<double, 15, 15>::Identity()};
    bool whiten_{false};
};

/* ============================================================================
 * 因子 2：GNSS 位置（3 维）
 * ==========================================================================*/
struct GnssPosResidual {
    GnssPosResidual(const Eigen::Vector3d &pos, const Eigen::Vector3d &std_dev)
        : pos_(pos), std_(std_dev) {}

    template <typename T>
    bool operator()(const T *const pose, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(pose);
        Eigen::Map<Eigen::Matrix<T, 3, 1>>       r(residual);
        const Eigen::Matrix<T, 3, 1> pos = toEig3<T>(pos_);
        const Eigen::Matrix<T, 3, 1> sd  = toEig3<T>(std_);
        for (int i = 0; i < 3; ++i) r(i) = (p(i) - pos(i)) / sd(i);
        return true;
    }

    Eigen::Vector3d pos_, std_;
};


/* ============================================================================
 * 因子 3：GNSS 速度（3 维，增强版）
 *
 *   观测：vn（ENU，m/s）
 *   残差：r_i = (v_i - vn_i) / sigma_i，可加 Huber
 * ==========================================================================*/
struct GnssVelResidual {
    GnssVelResidual(const Eigen::Vector3d &vn, const Eigen::Vector3d &std_dev, double huber_delta = 0.0)
        : vn_(vn), std_(std_dev), huber_delta_(huber_delta) {}

    template <typename T>
    bool operator()(const T *const mix, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v(mix);
        Eigen::Map<Eigen::Matrix<T, 3, 1>>       r(residual);
        const Eigen::Matrix<T, 3, 1> vn = toEig3<T>(vn_);
        const Eigen::Matrix<T, 3, 1> sd = toEig3<T>(std_);

        for (int i = 0; i < 3; ++i) {
            const T s = (sd(i) > T(1e-9)) ? sd(i) : T(1e-9);
            r(i) = (v(i) - vn(i)) / s;
        }

        if (huber_delta_ > 0.0) {
            for (int i = 0; i < 3; ++i) {
                const T abs_r = ceres::abs(r(i));
                if (abs_r > T(huber_delta_)) {
                    const T sign_r = (r(i) >= T(0)) ? T(1) : T(-1);
                    r(i) = sign_r * ceres::sqrt(
                               T(2.0) * T(huber_delta_) * abs_r
                               - T(huber_delta_) * T(huber_delta_));
                }
            }
        }
        return true;
    }

    Eigen::Vector3d vn_, std_;
    double huber_delta_;
};

/* ============================================================================
 * 因子 4：GNSS 航向（1 维，含航向偏置标定）
 *
 *   观测 yaw_gnss（已做 ant_mode 折算，但不含 yaw_offset）
 *   残差  r = wrap(yaw(q) + yaw_off - yaw_gnss) / sigma
 *   yaw_off 从标定参数块 calib[6] 读取
 * ==========================================================================*/
struct GnssYawResidual {
    GnssYawResidual(double yaw_gnss, double std_dev, double huber_delta = 0.0)
        : yaw_gnss_(yaw_gnss),
          std_(std_dev > 1e-9 ? std_dev : 1e-9),
          huber_delta_(huber_delta) {}

    template <typename T>
    bool operator()(const T *const pose, const T *const calib, T *residual) const {
        Eigen::Quaternion<T> q(pose[6], pose[3], pose[4], pose[5]);
        Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();

        const T yaw_est = ceres::atan2(-R(0, 1), R(1, 1));  /* 与 m2att 一致 */
        const T yaw_off = calib[6];                          /* 标定量 */

        const T d_raw = yaw_est + yaw_off - T(yaw_gnss_);
        const T d     = ceres::atan2(ceres::sin(d_raw), ceres::cos(d_raw));

        T r = d / T(std_);
        if (huber_delta_ > 0.0) {
            const T abs_r = ceres::abs(r);
            if (abs_r > T(huber_delta_)) {
                const T sign_r = (r >= T(0)) ? T(1) : T(-1);
                r = sign_r * ceres::sqrt(
                        T(2.0) * T(huber_delta_) * abs_r
                        - T(huber_delta_) * T(huber_delta_));
            }
        }
        residual[0] = r;
        return true;
    }

    double yaw_gnss_, std_, huber_delta_;
};

/* ============================================================================
 * 因子 5：静止零速（3 维，增强版）
 *
 *   观测：v = 0
 *   残差：r_i = v_i / sigma，可加 Huber
 * ==========================================================================*/
struct StaticVelResidual {
    explicit StaticVelResidual(double sigma, double huber_delta = 0.0)
        : sigma_(sigma > 1e-9 ? sigma : 1e-9),
          huber_delta_(huber_delta) {}

    template <typename T>
    bool operator()(const T *const mix, T *residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v(mix);
        for (int i = 0; i < 3; ++i) {
            T r = v(i) / T(sigma_);
            if (huber_delta_ > 0.0) {
                const T abs_r = ceres::abs(r);
                if (abs_r > T(huber_delta_)) {
                    const T sign_r = (r >= T(0)) ? T(1) : T(-1);
                    r = sign_r * ceres::sqrt(
                            T(2.0) * T(huber_delta_) * abs_r
                            - T(huber_delta_) * T(huber_delta_));
                }
            }
            residual[i] = r;
        }
        return true;
    }

    double sigma_;
    double huber_delta_;
};

/* ============================================================================
 * 边缘化先验（多参数块 + 流形感知）
 *
 * 残差：      ρ(δ) = J_out · δ + r_out
 *             其中 δ 是各保留参数块**相对线性化点**的切空间增量：
 *               POSE   : δ = Log(x_lin⁻¹ ⊗ x)        （PoseManifold::Minus）
 *               CALIB  : δ = CalibManifold::Minus(x, x_lin)
 *               ADDITIVE: δ = x − x_lin               （mix、模糊度等纯加性块）
 * 雅可比：    ∂ρ/∂ambient_i = J_out[:, off_i : off_i+t_i] · MinusJacobian_i(x_i)
 *
 * 符号：本工程法方程约定 b = Jᵀr，schur_complement 保持该约定输出
 *       J_outᵀ r_out = b*。Ceres 因子 ρ 对 b 的贡献是 Jᵀρ(0)，
 *       要等于 b* 就必须 ρ(0) = r_out ⇒ ρ(δ) = J·δ + r_out（加号）。
 *
 * 与旧版的区别：旧版把 J.clm 维当成**一个** ambient 参数块，而 pose 是
 * ambient 7 / tangent 6（带流形），ambient 维数根本不等于切空间维数，
 * 因此永远挂不到 buildProblem 建出来的参数块上。这里按块分别声明与求导。
 * ==========================================================================*/
enum class MargBlockKind { ADDITIVE, POSE, CALIB };

struct MargPriorBlock {
    std::vector<double> lin;                        /*< 线性化点的 ambient 值副本 */
    int ambient{0};
    int tangent{0};
    MargBlockKind kind{MargBlockKind::ADDITIVE};
};

class MarginalizationPriorFactor : public ceres::CostFunction {
public:
    MarginalizationPriorFactor(const Eigen::MatrixXd &J, const Eigen::VectorXd &r,
                               std::vector<MargPriorBlock> blocks)
        : J_(J), r_(r), blocks_(std::move(blocks)) {
        set_num_residuals(static_cast<int>(J_.rows()));
        for (const auto &b : blocks_) {
            mutable_parameter_block_sizes()->push_back(b.ambient);
            offsets_.push_back(n_tan_);
            n_tan_ += b.tangent;
        }
    }

    bool Evaluate(const double *const *params, double *residuals,
                  double **jacobians) const override {
        const int nr = static_cast<int>(J_.rows());
        const int nc = static_cast<int>(J_.cols());
        if (nc != n_tan_) return false;

        Eigen::VectorXd dx(n_tan_);
        for (size_t i = 0; i < blocks_.size(); ++i) {
            const MargPriorBlock &b = blocks_[i];
            const double *x = params[i];
            const int off = offsets_[i];
            if (b.kind == MargBlockKind::POSE) {
                double dphi[6];
                pose_.Minus(x, b.lin.data(), dphi);
                for (int k2 = 0; k2 < 6; ++k2) dx(off + k2) = dphi[k2];
            } else if (b.kind == MargBlockKind::CALIB) {
                double d[13];
                calib_.Minus(x, b.lin.data(), d);
                for (int k2 = 0; k2 < 13; ++k2) dx(off + k2) = d[k2];
            } else {
                for (int k2 = 0; k2 < b.tangent; ++k2) dx(off + k2) = x[k2] - b.lin[k2];
            }
        }

        Eigen::Map<Eigen::VectorXd> res(residuals, nr);
        res.noalias() = J_ * dx + r_;

        if (jacobians) {
            for (size_t i = 0; i < blocks_.size(); ++i) {
                if (!jacobians[i]) continue;
                const MargPriorBlock &b = blocks_[i];
                const int off = offsets_[i];
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>> Ji(jacobians[i], nr, b.ambient);
                const auto Jblk = J_.block(0, off, nr, b.tangent);
                if (b.kind == MargBlockKind::POSE) {
                    Eigen::Matrix<double, 6, 7, Eigen::RowMajor> M;
                    pose_.MinusJacobian(params[i], M.data());
                    Ji.noalias() = Jblk * M;
                } else if (b.kind == MargBlockKind::CALIB) {
                    Eigen::Matrix<double, 13, 14, Eigen::RowMajor> M;
                    calib_.MinusJacobian(params[i], M.data());
                    Ji.noalias() = Jblk * M;
                } else {
                    Ji = Jblk;
                }
            }
        }
        return true;
    }

    int tangentSize(void) const { return n_tan_; }

private:
    Eigen::MatrixXd J_;
    Eigen::VectorXd r_;
    std::vector<MargPriorBlock> blocks_;
    std::vector<int> offsets_;
    int n_tan_{0};
    PoseManifold  pose_;    /*< 仅用于求 Minus / MinusJacobian（无状态） */
    CalibManifold calib_;
};

/* ============================================================================
 * 因子 7：里程计速度（3 维，含刻度因子 / 安装角 / 杆臂标定）
 *
 *   v_odo^b = (1 + sodo) * (dS/dt) * C_b^m * e_x + omega_ib^b × l_OD
 *   r = R_b^n * v_odo^b - v^n
 *
 *   标定量从 calib 块读取：
 *     calib[0]     = sodo
 *     calib[1]     = abv_pitch
 *     calib[2]     = abv_yaw
 *     calib[3..5]  = lvOD
 *     calib[6]     = yaw_off（本因子不使用）
 * ==========================================================================*/
struct OdoVelResidual {
    OdoVelResidual(double dS, double dt, const Eigen::Vector3d &omega_meas,
                   double sigma, double huber_delta = 0.0)
        : dS_(dS),
          dt_(dt > 1e-9 ? dt : 1e-9),
          omega_meas_(omega_meas),
          sigma_(sigma > 1e-9 ? sigma : 1e-9),
          huber_delta_(huber_delta) {}

    template <typename T>
    bool operator()(const T *const pose, const T *const mix,
                    const T *const calib, T *residual) const {
        Eigen::Quaternion<T> q(pose[6], pose[3], pose[4], pose[5]);
        Eigen::Matrix<T, 3, 3> R_bn = q.toRotationMatrix();

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_n(mix);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg(mix + 3);

        /* --- 从标定块读参数 --- */
        const T sodo = calib[0];
        const T ap   = calib[1];
        const T ay   = calib[2];
        Eigen::Matrix<T, 3, 1> l(calib[3], calib[4], calib[5]);

        /* --- 安装角：C_b^m * e_x --- */
        Eigen::Matrix<T, 3, 1> cx;
        cx <<  ceres::cos(ay) * ceres::cos(ap),
              -ceres::sin(ay) * ceres::cos(ap),
              -ceres::sin(ap);

        /* --- 杆臂：omega_ib^b × l_OD --- */
        Eigen::Matrix<T, 3, 1> w(
            T(omega_meas_.x()) - bg(0),
            T(omega_meas_.y()) - bg(1),
            T(omega_meas_.z()) - bg(2));
        Eigen::Matrix<T, 3, 1> v_lev = w.cross(l);

        /* --- 里程计速度（b 系） --- */
        Eigen::Matrix<T, 3, 1> v_odo_b =
            (T(1.0) + sodo) * T(dS_ / dt_) * cx + v_lev;

        /* --- 转 n 系并求残差 --- */
        Eigen::Matrix<T, 3, 1> v_odo_n = R_bn * v_odo_b;
        Eigen::Map<Eigen::Matrix<T, 3, 1>> r(residual);
        r = (v_n - v_odo_n) / T(sigma_);

        if (huber_delta_ > 0.0) {
            for (int i = 0; i < 3; ++i) {
                const T abs_r = ceres::abs(r(i));
                if (abs_r > T(huber_delta_)) {
                    const T sign_r = (r(i) >= T(0)) ? T(1) : T(-1);
                    r(i) = sign_r * ceres::sqrt(
                               T(2.0) * T(huber_delta_) * abs_r
                               - T(huber_delta_) * T(huber_delta_));
                }
            }
        }
        return true;
    }

    double dS_, dt_;
    Eigen::Vector3d  omega_meas_;
    double sigma_, huber_delta_;
};

/* ============================================================================
 * 因子 8：视觉重投影（2 维，手写雅可比，纯 Matrix3d / Vector3d 运算）
 *
 * 参数块顺序：pose_i[7] · pose_j[7] · calib[14] · inv_depth[1]
 *
 * 残差：r = π(T_cj_w · T_wc_i · (obs_i / ρ)) - obs_j
 *
 * 雅可比输出的是"对 ambient 参数的偏导"：
 *   pose 的 q 部分用 ∂r/∂q = 2·(∂r/∂φ)·Q(q)^T 转换（详见下文），
 *   Ceres 的 PoseManifold 会自动把结果投影到 6 维切空间。
 * ==========================================================================*/
struct VisualReprojResidual : public ceres::CostFunction {
    VisualReprojResidual(const Eigen::Vector3d& obs_i, const Eigen::Vector3d& obs_j, double sigma)
        : obs_i_(obs_i), obs_j_(obs_j),
          sigma_inv_(1.0 / (sigma > 1e-9 ? sigma : 1e-9)) {
        set_num_residuals(2);
        mutable_parameter_block_sizes()->push_back(7);          /* pose_i */
        mutable_parameter_block_sizes()->push_back(7);          /* pose_j */
        mutable_parameter_block_sizes()->push_back(NUM_CALIB);  /* calib  */
        mutable_parameter_block_sizes()->push_back(1);          /* inv_depth */
    }

    bool Evaluate(const double* const* params,
                  double* residuals,
                  double** jacobians) const override {
        /* ---------- 1) 拆参数 ---------- */
        const double* pose_i = params[0];
        const double* pose_j = params[1];
        const double* calib  = params[2];
        const double  rho    = params[3][0];

        const Eigen::Vector3d p_i(pose_i[0], pose_i[1], pose_i[2]);
        const Eigen::Quaterniond  q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        const Eigen::Vector3d p_j(pose_j[0], pose_j[1], pose_j[2]);
        const Eigen::Quaterniond  q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);

        const Eigen::Vector3d t_bc(calib[7], calib[8], calib[9]);
        const Eigen::Quaterniond  q_bc(calib[13], calib[10], calib[11], calib[12]);

        /* ---------- 2) 旋转矩阵 ---------- */
        const Eigen::Matrix3d R_i  = q2mat(q_i);
        const Eigen::Matrix3d R_j  = q2mat(q_j);
        const Eigen::Matrix3d R_bc = q2mat(q_bc);

        /* ---------- 3) T_wc_i / T_wc_j ---------- */
        const Eigen::Matrix3d  R_wc_i = R_i * R_bc;
        const Eigen::Vector3d p_wc_i = p_i + R_i * t_bc;    /* Eigen::Matrix3d * Eigen::Vector3d */

        const Eigen::Matrix3d  R_wc_j = R_j * R_bc;
        const Eigen::Vector3d p_wc_j = p_j + R_j * t_bc;

        /* ---------- 4) 特征点在相机 i 系下的 3D 坐标 ---------- */
        const double inv_rho = 1.0 / rho;
        const Eigen::Vector3d  P_c_i(obs_i_.x() * inv_rho,
                           obs_i_.y() * inv_rho,
                           inv_rho);

        /* ---------- 5) 转世界系，再转相机 j 系 ---------- */
        const Eigen::Vector3d P_w   = p_wc_i + R_wc_i * P_c_i;
        const Eigen::Vector3d P_c_j = ((R_wc_j.transpose())) * (P_w - p_wc_j);

        /* ---------- 6) 残差 ---------- */
        double Z = P_c_j.z();
        /* 深度钳位：必须是"单侧"钳位。
         * 原先用 std::fabs(Z) < 1e-6 会把 Z∈(−1e-6,0) 翻成 +1e-6 —— 深度符号
         * 被翻转，投影变成镜像的"看起来合理"的残差，会把 BA 拉向错误解。
         * 这里改成 Z<=1e-6（含 NaN）一律钳到 1e-6：不翻符号、不给 NaN，
         * 负深度会得到很大的残差，从而把点推回相机前方。
         * （更彻底的做法是在上游按 cheirality 直接剔除该观测，
         *   但那需要改动 VisualObs 的生产方，留作后续工作。） */
        if (!(Z > 1e-6)) Z = 1e-6;
        const double Zinv = 1.0 / Z;
        const double X_Z  = P_c_j.x() * Zinv;
        const double Y_Z  = P_c_j.y() * Zinv;

        residuals[0] = (X_Z - obs_j_.x()) * sigma_inv_;
        residuals[1] = (Y_Z - obs_j_.y()) * sigma_inv_;

        if (jacobians == nullptr) return true;

        /* ---------- 7) 投影函数的雅可比：用两行 Vector3d 表示 ---------- */
        const Eigen::Vector3d J_pi_0(Zinv, 0.0, -X_Z * Zinv);
        const Eigen::Vector3d J_pi_1(0.0, Zinv, -Y_Z * Zinv);

        const Eigen::Matrix3d R_wc_j_T = (R_wc_j.transpose());

        /* ---------- 8) 对 pose_i 的雅可比（2×7 ambient） ---------- */
        if (jacobians[0] != nullptr) {
            /* 特征在 body_i 系下的坐标 */
            const Eigen::Vector3d P_i_Bi = t_bc + R_bc * P_c_i;

            /* ∂r/∂p_i = σ⁻¹ · J_pi · R_wc_j^T           (2×3) */
            /* 旧 vect3 * Matrix3d 语义是行向量右乘 vᵀM，等价写成 (Mᵀ·v) */
            const Eigen::Vector3d dpi_p0 = R_wc_j_T.transpose() * J_pi_0;
            const Eigen::Vector3d dpi_p1 = R_wc_j_T.transpose() * J_pi_1;

            /* ∂r/∂φ_i = -σ⁻¹ · J_pi · R_wc_j^T · R_i · [P_i_Bi]_×  (2×3) */
            const Eigen::Matrix3d tmp     = R_wc_j_T * R_i * askew(P_i_Bi);
            const Eigen::Vector3d dpi_phi0 = tmp.transpose() * J_pi_0;
            const Eigen::Vector3d dpi_phi1 = tmp.transpose() * J_pi_1;

            /* 位置 3 列 */
            jacobians[0][0 * 7 + 0] = dpi_p0.x() * sigma_inv_;
            jacobians[0][0 * 7 + 1] = dpi_p0.y() * sigma_inv_;
            jacobians[0][0 * 7 + 2] = dpi_p0.z() * sigma_inv_;
            jacobians[0][1 * 7 + 0] = dpi_p1.x() * sigma_inv_;
            jacobians[0][1 * 7 + 1] = dpi_p1.y() * sigma_inv_;
            jacobians[0][1 * 7 + 2] = dpi_p1.z() * sigma_inv_;

            /* ∂r/∂q = 2·(∂r/∂δφ)·Q(q)^T，Q(q)^T 的 4 列按 [qx, qy, qz, qw] 排列：
            *   col(qx) = ( qw, -qz,  qy)
            *   col(qy) = ( qz,  qw, -qx)
            *   col(qz) = (-qy,  qx,  qw)
            *   col(qw) = (-qx, -qy, -qz) */
            const double qx = q_i.x(), qy = q_i.y(), qz = q_i.z(), qw = q_i.w();
            const Eigen::Vector3d dphi_r0(-dpi_phi0.x(), -dpi_phi0.y(), -dpi_phi0.z());
            const Eigen::Vector3d dphi_r1(-dpi_phi1.x(), -dpi_phi1.y(), -dpi_phi1.z());
            const Eigen::Vector3d col_qx( qw, -qz,  qy);
            const Eigen::Vector3d col_qy( qz,  qw, -qx);
            const Eigen::Vector3d col_qz(-qy,  qx,  qw);
            const Eigen::Vector3d col_qw(-qx, -qy, -qz);

            jacobians[0][0*7+3] = 2.0 * dot(dphi_r0, col_qx) * sigma_inv_;
            jacobians[0][0*7+4] = 2.0 * dot(dphi_r0, col_qy) * sigma_inv_;
            jacobians[0][0*7+5] = 2.0 * dot(dphi_r0, col_qz) * sigma_inv_;
            jacobians[0][0*7+6] = 2.0 * dot(dphi_r0, col_qw) * sigma_inv_;
            jacobians[0][1*7+3] = 2.0 * dot(dphi_r1, col_qx) * sigma_inv_;
            jacobians[0][1*7+4] = 2.0 * dot(dphi_r1, col_qy) * sigma_inv_;
            jacobians[0][1*7+5] = 2.0 * dot(dphi_r1, col_qz) * sigma_inv_;
            jacobians[0][1*7+6] = 2.0 * dot(dphi_r1, col_qw) * sigma_inv_;
        }

        /* ---------- 9) 对 pose_j 的雅可比（2×7 ambient） ---------- */
        if (jacobians[1] != nullptr) {
            /* 特征在 body_j 系下的坐标 */
            const Eigen::Vector3d P_Bj = ((R_j.transpose())) * (P_w - p_j);

            /* ∂r/∂p_j = -σ⁻¹ · J_pi · R_wc_j^T */
            const Eigen::Vector3d dpj_p0 = R_wc_j_T.transpose() * J_pi_0;
            const Eigen::Vector3d dpj_p1 = R_wc_j_T.transpose() * J_pi_1;

            /* ∂r/∂φ_j = σ⁻¹ · J_pi · R_bc^T · [P_Bj]_× */
            const Eigen::Matrix3d tmp     = ((R_bc.transpose())) * askew(P_Bj);
            const Eigen::Vector3d dpj_phi0 = tmp.transpose() * J_pi_0;
            const Eigen::Vector3d dpj_phi1 = tmp.transpose() * J_pi_1;

            jacobians[1][0*7+0] = -dpj_p0.x() * sigma_inv_;
            jacobians[1][0*7+1] = -dpj_p0.y() * sigma_inv_;
            jacobians[1][0*7+2] = -dpj_p0.z() * sigma_inv_;
            jacobians[1][1*7+0] = -dpj_p1.x() * sigma_inv_;
            jacobians[1][1*7+1] = -dpj_p1.y() * sigma_inv_;
            jacobians[1][1*7+2] = -dpj_p1.z() * sigma_inv_;

            /* ∂r/∂q_j = 2·(∂r/∂δφ_j)·Q(q_j)^T
            * Q(q_j)^T 的 4 列按 [qx, qy, qz, qw]：
            *   col(qx) = ( qw, -qz,  qy)
            *   col(qy) = ( qz,  qw, -qx)
            *   col(qz) = (-qy,  qx,  qw)
            *   col(qw) = (-qx, -qy, -qz)  */
            const double qx = q_j.x(), qy = q_j.y(), qz = q_j.z(), qw = q_j.w();
            const Eigen::Vector3d col_qx( qw, -qz,  qy);
            const Eigen::Vector3d col_qy( qz,  qw, -qx);
            const Eigen::Vector3d col_qz(-qy,  qx,  qw);
            const Eigen::Vector3d col_qw(-qx, -qy, -qz);
            const Eigen::Vector3d dr0(dpj_phi0.x(), dpj_phi0.y(), dpj_phi0.z());
            const Eigen::Vector3d dr1(dpj_phi1.x(), dpj_phi1.y(), dpj_phi1.z());

            jacobians[1][0*7+3] = 2.0 * dot(dr0, col_qx) * sigma_inv_;
            jacobians[1][0*7+4] = 2.0 * dot(dr0, col_qy) * sigma_inv_;
            jacobians[1][0*7+5] = 2.0 * dot(dr0, col_qz) * sigma_inv_;
            jacobians[1][0*7+6] = 2.0 * dot(dr0, col_qw) * sigma_inv_;
            jacobians[1][1*7+3] = 2.0 * dot(dr1, col_qx) * sigma_inv_;
            jacobians[1][1*7+4] = 2.0 * dot(dr1, col_qy) * sigma_inv_;
            jacobians[1][1*7+5] = 2.0 * dot(dr1, col_qz) * sigma_inv_;
            jacobians[1][1*7+6] = 2.0 * dot(dr1, col_qw) * sigma_inv_;
        }

        /* ---------- 10) 对 calib[14] 的雅可比 ---------- */
        if (jacobians[2] != nullptr) {
            /* 全部清零（calib[0..6] 的列全为 0） */
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < NUM_CALIB; ++c)
                    jacobians[2][r * NUM_CALIB + c] = 0.0;

            /* --- 对 t_bc（calib[7..9]）--- */
            /* ∂r/∂t_bc = σ⁻¹ · J_pi · (R_wc_j^T · R_i - R_bc^T)
            * 注意：δt 在机体系（CalibManifold::Plus 是纯加性扰动），
            *       所以是 R_bc^T，不是 I_3（那是相机系扰动的结果）。 */
            const Eigen::Matrix3d Rd = R_wc_j_T * R_i - ((R_bc.transpose()));

            const Eigen::Vector3d dt_0 = Rd.transpose() * J_pi_0;
            const Eigen::Vector3d dt_1 = Rd.transpose() * J_pi_1;

            jacobians[2][0*NUM_CALIB + 7] = dt_0.x() * sigma_inv_;
            jacobians[2][0*NUM_CALIB + 8] = dt_0.y() * sigma_inv_;
            jacobians[2][0*NUM_CALIB + 9] = dt_0.z() * sigma_inv_;
            jacobians[2][1*NUM_CALIB + 7] = dt_1.x() * sigma_inv_;
            jacobians[2][1*NUM_CALIB + 8] = dt_1.y() * sigma_inv_;
            jacobians[2][1*NUM_CALIB + 9] = dt_1.z() * sigma_inv_;

            /* --- 对 q_bc（calib[10..13]）---
             * ∂r/∂φ_bc = σ⁻¹ · J_pi · ([P_c_j]_× - R_wc_j^T · R_wc_i · [P_c_i]_×)
             */
            const Eigen::Matrix3d inner = R_wc_j_T * R_wc_i * askew(P_c_i);
            const Eigen::Matrix3d outer = askew(P_c_j) - inner;

            const Eigen::Vector3d dphi_0 = outer.transpose() * J_pi_0;
            const Eigen::Vector3d dphi_1 = outer.transpose() * J_pi_1;

            const double qbx = q_bc.x(), qby = q_bc.y(),
                        qbz = q_bc.z(), qbw = q_bc.w();
            const Eigen::Vector3d col_qx( qbw, -qbz,  qby);
            const Eigen::Vector3d col_qy( qbz,  qbw, -qbx);
            const Eigen::Vector3d col_qz(-qby,  qbx,  qbw);
            const Eigen::Vector3d col_qw(-qbx, -qby, -qbz);
            const Eigen::Vector3d dr0(dphi_0.x(), dphi_0.y(), dphi_0.z());
            const Eigen::Vector3d dr1(dphi_1.x(), dphi_1.y(), dphi_1.z());

            jacobians[2][0*NUM_CALIB + 10] = 2.0 * dot(dr0, col_qx) * sigma_inv_;
            jacobians[2][0*NUM_CALIB + 11] = 2.0 * dot(dr0, col_qy) * sigma_inv_;
            jacobians[2][0*NUM_CALIB + 12] = 2.0 * dot(dr0, col_qz) * sigma_inv_;
            jacobians[2][0*NUM_CALIB + 13] = 2.0 * dot(dr0, col_qw) * sigma_inv_;
            jacobians[2][1*NUM_CALIB + 10] = 2.0 * dot(dr1, col_qx) * sigma_inv_;
            jacobians[2][1*NUM_CALIB + 11] = 2.0 * dot(dr1, col_qy) * sigma_inv_;
            jacobians[2][1*NUM_CALIB + 12] = 2.0 * dot(dr1, col_qz) * sigma_inv_;
            jacobians[2][1*NUM_CALIB + 13] = 2.0 * dot(dr1, col_qw) * sigma_inv_;
        }

        /* ---------- 11) 对逆深度的雅可比（2×1） ---------- */
        if (jacobians[3] != nullptr) {
            /* ∂r/∂ρ = -σ⁻¹/ρ · J_pi · R_wc_j^T · R_wc_i · P_c_i
            * 注意 J_pi_k 是 3D 行向量，右侧 (R_wc_j^T·R_wc_i·P_c_i) 是 3D 列向量，
            * 二者相乘是点乘（不是叉乘） */
            const Eigen::Vector3d u = R_wc_j_T * (R_wc_i * P_c_i);   /* 3D 向量 */
            const double dr0 = dot(J_pi_0, u);
            const double dr1 = dot(J_pi_1, u);
            jacobians[3][0] = -dr0 * sigma_inv_ / rho;
            jacobians[3][1] = -dr1 * sigma_inv_ / rho;
        }

        return true;
    }

    Eigen::Vector3d  obs_i_, obs_j_;
    double sigma_inv_;
};

/* ============================================================================
 * 因子：双差伪距（单频点，1 维，手推雅可比）
 *
 * 参数块：pose[7]
 * 残差：r = (dd_P[f] - dd_ρ) / σ_f
 *
 * 雅可比：
 *   ∂r/∂p_n = (κ_i - κ_k)ᵀ · R_en / σ_f
 *   ∂r/∂δφ  = (κ_i - κ_k)ᵀ · R_en · R_bn · (-[t_ant]×) / σ_f
 * ==========================================================================*/
struct GnssDDPseudorangeFreqResidual : public ceres::CostFunction {
    GnssDDPseudorangeFreqResidual(const DDObs& dd, int freq,
                                  const Eigen::Vector3d& base_pos,
                                  const Eigen::Matrix3d&  R_en,
                                  const Eigen::Vector3d& p0,
                                  const Eigen::Vector3d& t_ant)
        : dd_(dd), freq_(freq), base_pos_(base_pos),
          R_en_(R_en), p0_(p0), t_ant_(t_ant) {
        const double sigma = std::max(GPS_FREQ[freq].sigma_P * 1.4142, 0.1);
        sigma_inv_ = 1.0 / sigma;
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(7);
    }

    bool Evaluate(const double* const* params,
                  double* residuals, double** jacobians) const override {
        const double* pose = params[0];

        const Eigen::Vector3d p_n(pose[0], pose[1], pose[2]);
        const Eigen::Quaterniond  q  (pose[6], pose[3], pose[4], pose[5]);
        const Eigen::Matrix3d  R_bn = q2mat(q);

        const Eigen::Vector3d p_e = p0_ + R_en_ * (p_n + R_bn * t_ant_);

        const Eigen::Vector3d si = dd_.sat_i.pos_ecef - p_e;
        const Eigen::Vector3d sk = dd_.sat_k.pos_ecef - p_e;
        const double rho_i = norm(si);
        const double rho_k = norm(sk);
        if (rho_i < 1e-6 || rho_k < 1e-6) { residuals[0] = 0.0; return true; }

        const Eigen::Vector3d bi = dd_.sat_i.pos_ecef - base_pos_;
        const Eigen::Vector3d bk = dd_.sat_k.pos_ecef - base_pos_;
        const double dd_rho = (rho_i - rho_k) - (norm(bi) - norm(bk));

        residuals[0] = (dd_.dd_P[freq_] - dd_rho) * sigma_inv_;

        if (jacobians == nullptr) return true;

        const Eigen::Vector3d kappa_i = si / rho_i;
        const Eigen::Vector3d kappa_k = sk / rho_k;
        const Eigen::Vector3d dr_dpe  = (kappa_i - kappa_k) * sigma_inv_;

        if (jacobians[0] != nullptr) {
            const Eigen::Vector3d dr_dpn = R_en_.transpose() * dr_dpe;
            jacobians[0][0] = dr_dpn.x();
            jacobians[0][1] = dr_dpn.y();
            jacobians[0][2] = dr_dpn.z();

            const Eigen::Vector3d dr_dphi =
                ((R_en_ * R_bn) * (-askew(t_ant_))).transpose() * dr_dpe;
            projPhiToQAmbient(dr_dphi, q, &jacobians[0][3]);
        }
        return true;
    }

    DDObs  dd_;
    int    freq_;
    Eigen::Vector3d  base_pos_, p0_, t_ant_;
    Eigen::Matrix3d   R_en_;
    double sigma_inv_;
};

/* ============================================================================
 * 因子：双差载波（单频点，1 维）
 * 参数块：pose[7] · amb[1]（cycle）
 * 残差：r = (λ_f · dd_L[f] - dd_ρ - λ_f · N_f) / σ_f
 * 雅可比：对 pose 与伪距相同；对 amb = -λ_f / σ_f
 * ==========================================================================*/
struct GnssDDCarrierFreqResidual : public ceres::CostFunction {
    GnssDDCarrierFreqResidual(const DDObs& dd, int freq,
                              const Eigen::Vector3d& base_pos,
                              const Eigen::Matrix3d&  R_en,
                              const Eigen::Vector3d& p0,
                              const Eigen::Vector3d& t_ant)
        : dd_(dd), freq_(freq), base_pos_(base_pos),
          R_en_(R_en), p0_(p0), t_ant_(t_ant) {
        const double sigma = std::max(GPS_FREQ[freq].sigma_L * 1.4142, 1e-4);
        sigma_inv_ = 1.0 / sigma;
        lambda_    = GPS_FREQ[freq].lambda;
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(7);
        mutable_parameter_block_sizes()->push_back(1);
    }

    bool Evaluate(const double* const* params,
                  double* residuals, double** jacobians) const override {
        const double* pose = params[0];
        const double* amb  = params[1];

        const Eigen::Vector3d p_n(pose[0], pose[1], pose[2]);
        const Eigen::Quaterniond  q  (pose[6], pose[3], pose[4], pose[5]);
        const Eigen::Matrix3d  R_bn = q2mat(q);
        const Eigen::Vector3d p_e  = p0_ + R_en_ * (p_n + R_bn * t_ant_);

        const Eigen::Vector3d si = dd_.sat_i.pos_ecef - p_e;
        const Eigen::Vector3d sk = dd_.sat_k.pos_ecef - p_e;
        const double rho_i = norm(si);
        const double rho_k = norm(sk);
        if (rho_i < 1e-6 || rho_k < 1e-6) { residuals[0] = 0.0; return true; }

        const Eigen::Vector3d bi = dd_.sat_i.pos_ecef - base_pos_;
        const Eigen::Vector3d bk = dd_.sat_k.pos_ecef - base_pos_;
        const double dd_rho = (rho_i - rho_k) - (norm(bi) - norm(bk));

        residuals[0] = (lambda_ * dd_.dd_L[freq_] - dd_rho - lambda_ * amb[0])
                       * sigma_inv_;

        if (jacobians == nullptr) return true;

        const Eigen::Vector3d kappa_i = si / rho_i;
        const Eigen::Vector3d kappa_k = sk / rho_k;
        const Eigen::Vector3d dr_dpe  = (kappa_i - kappa_k) * sigma_inv_;

        if (jacobians[0] != nullptr) {
            const Eigen::Vector3d dr_dpn = R_en_.transpose() * dr_dpe;
            jacobians[0][0] = dr_dpn.x();
            jacobians[0][1] = dr_dpn.y();
            jacobians[0][2] = dr_dpn.z();

            const Eigen::Vector3d dr_dphi =
                ((R_en_ * R_bn) * (-askew(t_ant_))).transpose() * dr_dpe;
            projPhiToQAmbient(dr_dphi, q, &jacobians[0][3]);
        }
        if (jacobians[1] != nullptr) {
            jacobians[1][0] = -lambda_ * sigma_inv_;
        }
        return true;
    }

    DDObs  dd_;
    int    freq_;
    Eigen::Vector3d  base_pos_, p0_, t_ant_;
    Eigen::Matrix3d   R_en_;
    double sigma_inv_;
    double lambda_;
};

/* ============================================================================
 * 因子：双差宽巷 WL（1 维）
 * dd_L_WL = (f1·dd_L1 - f2·dd_L2) / (f1 - f2)  (cycle)
 * 残差：r = (λ_WL·dd_L_WL - dd_ρ - λ_WL·N_WL) / σ_WL
 * ==========================================================================*/
struct GnssDDWideLaneResidual : public ceres::CostFunction {
    GnssDDWideLaneResidual(const DDObs& dd, const Eigen::Vector3d& base_pos,
                           const Eigen::Matrix3d& R_en, const Eigen::Vector3d& p0,
                           const Eigen::Vector3d& t_ant, double sigma_wl)
        : dd_(dd), base_pos_(base_pos), R_en_(R_en), p0_(p0), t_ant_(t_ant) {
        sigma_inv_ = 1.0 / std::max(sigma_wl, 0.01);
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(7);
        mutable_parameter_block_sizes()->push_back(1);
    }

    bool Evaluate(const double* const* params,
                  double* residuals, double** jacobians) const override {
        const double* pose = params[0];
        const double* amb  = params[1];

        const Eigen::Vector3d p_n(pose[0], pose[1], pose[2]);
        const Eigen::Quaterniond  q  (pose[6], pose[3], pose[4], pose[5]);
        const Eigen::Matrix3d  R_bn = q2mat(q);
        const Eigen::Vector3d p_e  = p0_ + R_en_ * (p_n + R_bn * t_ant_);

        const Eigen::Vector3d si = dd_.sat_i.pos_ecef - p_e;
        const Eigen::Vector3d sk = dd_.sat_k.pos_ecef - p_e;
        const double rho_i = norm(si);
        const double rho_k = norm(sk);
        if (rho_i < 1e-6 || rho_k < 1e-6) { residuals[0] = 0.0; return true; }

        const Eigen::Vector3d bi = dd_.sat_i.pos_ecef - base_pos_;
        const Eigen::Vector3d bk = dd_.sat_k.pos_ecef - base_pos_;
        const double dd_rho = (rho_i - rho_k) - (norm(bi) - norm(bk));

        /* 宽巷：dd_L 的单位是"周"，宽巷观测量就是 φ1 − φ2（单位：周），
         * 乘 λ_WL 后才得到米。原先写成 (f1·ddL1 − f2·ddL2)/(f1 − f2) 会把
         * "加权周数"当成米数用，几何项被放大 (f1+f2)/(f1−f2) ≈ 8.0588 倍。 */
        const double dd_L_WL = dd_.dd_L[0] - dd_.dd_L[1];

        residuals[0] = (LAMBDA_WL * dd_L_WL - dd_rho
                        - LAMBDA_WL * amb[0]) * sigma_inv_;

        if (jacobians == nullptr) return true;

        const Eigen::Vector3d kappa_i = si / rho_i;
        const Eigen::Vector3d kappa_k = sk / rho_k;
        const Eigen::Vector3d dr_dpe  = (kappa_i - kappa_k) * sigma_inv_;

        if (jacobians[0] != nullptr) {
            const Eigen::Vector3d dr_dpn = R_en_.transpose() * dr_dpe;
            jacobians[0][0] = dr_dpn.x();
            jacobians[0][1] = dr_dpn.y();
            jacobians[0][2] = dr_dpn.z();

            const Eigen::Vector3d dr_dphi =
                ((R_en_ * R_bn) * (-askew(t_ant_))).transpose() * dr_dpe;
            projPhiToQAmbient(dr_dphi, q, &jacobians[0][3]);
        }
        if (jacobians[1] != nullptr) {
            jacobians[1][0] = -LAMBDA_WL * sigma_inv_;
        }
        return true;
    }

    DDObs  dd_;
    Eigen::Vector3d  base_pos_, p0_, t_ant_;
    Eigen::Matrix3d   R_en_;
    double sigma_inv_;
};

/* ============================================================================
 * 因子：双差窄巷 NL（1 维）
 * dd_L_NL = (f1·dd_L1 + f2·dd_L2) / (f1 + f2)  (cycle)
 * 残差：r = (λ_NL·dd_L_NL - dd_ρ - λ_NL·N_NL) / σ_NL
 * ==========================================================================*/
struct GnssDDNarrowLaneResidual : public ceres::CostFunction {
    GnssDDNarrowLaneResidual(const DDObs& dd, const Eigen::Vector3d& base_pos,
                             const Eigen::Matrix3d& R_en, const Eigen::Vector3d& p0,
                             const Eigen::Vector3d& t_ant, double sigma_nl)
        : dd_(dd), base_pos_(base_pos), R_en_(R_en), p0_(p0), t_ant_(t_ant) {
        sigma_inv_ = 1.0 / std::max(sigma_nl, 1e-4);
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(7);
        mutable_parameter_block_sizes()->push_back(1);
    }

    bool Evaluate(const double* const* params,
                  double* residuals, double** jacobians) const override {
        const double* pose = params[0];
        const double* amb  = params[1];

        const Eigen::Vector3d p_n(pose[0], pose[1], pose[2]);
        const Eigen::Quaterniond  q  (pose[6], pose[3], pose[4], pose[5]);
        const Eigen::Matrix3d  R_bn = q2mat(q);
        const Eigen::Vector3d p_e  = p0_ + R_en_ * (p_n + R_bn * t_ant_);

        const Eigen::Vector3d si = dd_.sat_i.pos_ecef - p_e;
        const Eigen::Vector3d sk = dd_.sat_k.pos_ecef - p_e;
        const double rho_i = norm(si);
        const double rho_k = norm(sk);
        if (rho_i < 1e-6 || rho_k < 1e-6) { residuals[0] = 0.0; return true; }

        const Eigen::Vector3d bi = dd_.sat_i.pos_ecef - base_pos_;
        const Eigen::Vector3d bk = dd_.sat_k.pos_ecef - base_pos_;
        const double dd_rho = (rho_i - rho_k) - (norm(bi) - norm(bk));

        /* 窄巷：同宽巷，dd_L 是周，窄巷观测量为 φ1 + φ2（单位：周）。
         * 原先的 (f1·ddL1 + f2·ddL2)/(f1 + f2) 使几何项变成
         * (f1²+f2²)/(f1+f2)² ≈ 0.5077 倍。 */
        const double dd_L_NL = dd_.dd_L[0] + dd_.dd_L[1];

        residuals[0] = (LAMBDA_NL * dd_L_NL - dd_rho
                        - LAMBDA_NL * amb[0]) * sigma_inv_;

        if (jacobians == nullptr) return true;

        const Eigen::Vector3d kappa_i = si / rho_i;
        const Eigen::Vector3d kappa_k = sk / rho_k;
        const Eigen::Vector3d dr_dpe  = (kappa_i - kappa_k) * sigma_inv_;

        if (jacobians[0] != nullptr) {
            const Eigen::Vector3d dr_dpn = R_en_.transpose() * dr_dpe;
            jacobians[0][0] = dr_dpn.x();
            jacobians[0][1] = dr_dpn.y();
            jacobians[0][2] = dr_dpn.z();

            const Eigen::Vector3d dr_dphi =
                ((R_en_ * R_bn) * (-askew(t_ant_))).transpose() * dr_dpe;
            projPhiToQAmbient(dr_dphi, q, &jacobians[0][3]);
        }
        if (jacobians[1] != nullptr) {
            jacobians[1][0] = -LAMBDA_NL * sigma_inv_;
        }
        return true;
    }

    DDObs  dd_;
    Eigen::Vector3d  base_pos_, p0_, t_ant_;
    Eigen::Matrix3d   R_en_;
    double sigma_inv_;
};

/* ============================================================================
 * 因子：双差 IF 载波（1 维）
 * dd_L_IF = α·λ1·dd_L1 + β·λ2·dd_L2   (m)
 * 残差：r = (dd_L_IF - dd_ρ - N_IF) / σ_IF
 * ==========================================================================*/
struct GnssDDIFCarrierResidual : public ceres::CostFunction {
    GnssDDIFCarrierResidual(const DDObs& dd, const Eigen::Vector3d& base_pos,
                            const Eigen::Matrix3d& R_en, const Eigen::Vector3d& p0,
                            const Eigen::Vector3d& t_ant, double sigma_if)
        : dd_(dd), base_pos_(base_pos), R_en_(R_en), p0_(p0), t_ant_(t_ant) {
        sigma_inv_ = 1.0 / std::max(sigma_if, 1e-3);
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(7);
        mutable_parameter_block_sizes()->push_back(1);
    }

    bool Evaluate(const double* const* params,
                  double* residuals, double** jacobians) const override {
        const double* pose = params[0];
        const double* amb  = params[1];

        const Eigen::Vector3d p_n(pose[0], pose[1], pose[2]);
        const Eigen::Quaterniond  q  (pose[6], pose[3], pose[4], pose[5]);
        const Eigen::Matrix3d  R_bn = q2mat(q);
        const Eigen::Vector3d p_e  = p0_ + R_en_ * (p_n + R_bn * t_ant_);

        const Eigen::Vector3d si = dd_.sat_i.pos_ecef - p_e;
        const Eigen::Vector3d sk = dd_.sat_k.pos_ecef - p_e;
        const double rho_i = norm(si);
        const double rho_k = norm(sk);
        if (rho_i < 1e-6 || rho_k < 1e-6) { residuals[0] = 0.0; return true; }

        const Eigen::Vector3d bi = dd_.sat_i.pos_ecef - base_pos_;
        const Eigen::Vector3d bk = dd_.sat_k.pos_ecef - base_pos_;
        const double dd_rho = (rho_i - rho_k) - (norm(bi) - norm(bk));

        const double L1_m = GPS_FREQ[0].lambda * dd_.dd_L[0];
        const double L2_m = GPS_FREQ[1].lambda * dd_.dd_L[1];
        const double dd_L_IF = ALPHA_IF * L1_m + BETA_IF * L2_m;

        residuals[0] = (dd_L_IF - dd_rho - amb[0]) * sigma_inv_;

        if (jacobians == nullptr) return true;

        const Eigen::Vector3d kappa_i = si / rho_i;
        const Eigen::Vector3d kappa_k = sk / rho_k;
        const Eigen::Vector3d dr_dpe  = (kappa_i - kappa_k) * sigma_inv_;

        if (jacobians[0] != nullptr) {
            const Eigen::Vector3d dr_dpn = R_en_.transpose() * dr_dpe;
            jacobians[0][0] = dr_dpn.x();
            jacobians[0][1] = dr_dpn.y();
            jacobians[0][2] = dr_dpn.z();

            const Eigen::Vector3d dr_dphi =
                ((R_en_ * R_bn) * (-askew(t_ant_))).transpose() * dr_dpe;
            projPhiToQAmbient(dr_dphi, q, &jacobians[0][3]);
        }
        if (jacobians[1] != nullptr) {
            jacobians[1][0] = -sigma_inv_;
        }
        return true;
    }

    DDObs  dd_;
    Eigen::Vector3d  base_pos_, p0_, t_ant_;
    Eigen::Matrix3d   R_en_;
    double sigma_inv_;
};

/* ============================================================================
 * 因子：钟差随机游走（1 维）
 * 参数块：clk_prev[1] · drift_prev[1] · clk_curr[1]
 * 残差：r = (clk_curr - clk_prev - drift_prev·dt) / σ
 * ==========================================================================*/
struct ClockBiasRWFactor : public ceres::CostFunction {
    ClockBiasRWFactor(double dt, double sigma_m)
        : dt_(dt), sigma_inv_(1.0 / std::max(sigma_m, 0.1)) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
    }
    bool Evaluate(const double* const* params,
                  double* residuals, double** jacobians) const override {
        residuals[0] = (params[2][0] - params[0][0]
                        - params[1][0] * dt_) * sigma_inv_;
        if (jacobians == nullptr) return true;
        if (jacobians[0]) jacobians[0][0] = -sigma_inv_;
        if (jacobians[1]) jacobians[1][0] = -dt_ * sigma_inv_;
        if (jacobians[2]) jacobians[2][0] =  sigma_inv_;
        return true;
    }
    double dt_, sigma_inv_;
};

/* ============================================================================
 * 因子：钟漂随机游走（1 维）
 * 参数块：drift_prev[1] · drift_curr[1]
 * ==========================================================================*/
struct ClockDriftRWFactor : public ceres::CostFunction {
    explicit ClockDriftRWFactor(double sigma_mps)
        : sigma_inv_(1.0 / std::max(sigma_mps, 0.01)) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
    }
    bool Evaluate(const double* const* params,
                  double* residuals, double** jacobians) const override {
        residuals[0] = (params[1][0] - params[0][0]) * sigma_inv_;
        if (jacobians == nullptr) return true;
        if (jacobians[0]) jacobians[0][0] = -sigma_inv_;
        if (jacobians[1]) jacobians[1][0] =  sigma_inv_;
        return true;
    }
    double sigma_inv_;
};

/* ============================================================================
 * 因子：模糊度软固定（1 维）
 * 参数块：amb[1]
 * 残差：r = (amb - N_int) / σ_fix
 * ==========================================================================*/
struct AmbiguityFixFactor : public ceres::CostFunction {
    explicit AmbiguityFixFactor(double N_int, double sigma_fix = 1.0e-3)
        : N_int_(N_int), sigma_inv_(1.0 / sigma_fix) {
        set_num_residuals(1);
        mutable_parameter_block_sizes()->push_back(1);
    }
    bool Evaluate(const double* const* params,
                  double* residuals, double** jacobians) const override {
        residuals[0] = (params[0][0] - N_int_) * sigma_inv_;
        if (jacobians && jacobians[0]) jacobians[0][0] = sigma_inv_;
        return true;
    }
    double N_int_, sigma_inv_;
};


struct FrameCompare {
    double time = 0.0;

    /* ---- 位置误差（局部 NEU，m）---- */
    bool   has_gnss_pos = false;
    double dpe = 0.0, dpn = 0.0, dpu = 0.0;   /* 分量差（est - gnss） */
    double dp_xy = 0.0, dp_3d = 0.0;          /* 平面/三维模长 */

    /* ---- 速度误差（n 系，m/s）---- */
    bool   has_gnss_vel = false;
    double dve = 0.0, dvn = 0.0, dvu = 0.0;
    double dv_xy = 0.0, dv_3d = 0.0;

    /* ---- 航向误差（rad / deg）---- */
    bool   has_gnss_yaw = false;
    double dyaw_rad = 0.0;
    double dyaw_deg = 0.0;
};

struct RunnerOptions {
    double kf_dt         = 1.0;
    int    max_keyframes = 15;
    bool   use_gnss_vel  = true;
    bool   use_gnss_yaw  = true;
    bool   use_static    = true;
    double static_thresh_gyr = 0.5 * DEG * 0.005;
    double static_thresh_acc = 0.05 * 0.005;
    double gnss_pos_scale    = 1.0;
    double gnss_vel_scale    = 1.0;

    /* ---- 双天线航向的坐标约定 ------------------------------------------------
     * ipos3g 内部按 RFU 体系 + ENU 导航系建姿态，双天线输出的参考轴与机体
     * 轴向可能差 0 / ±90 / 180 度，取决于基线怎么装（原工程 GNSS2Ali）：
     *   FB_B(0)  基线前后、输出指车尾 -> 原样
     *   FB_F(1)  基线前后、输出指车头 -> 再转 180°
     *   LR_L(2)  基线左右、输出指左侧 -> 再转 +90°
     *   LR_R(3)  基线左右、输出指右侧 -> 再转 -90°
     *   ONE(4)   单天线，航向不可用
     * yaw_offset 是天线安装角（rad），等价于原工程的 conf_GNSSAgle。 */
    /* 默认 FB_F：与 ipos3g_cmake/src/nav.cpp 里这台设备的配置一致
     * （configPara.ant_mode = Ant_Mode_FB_F, conf_GNSSAgle = 0）。 */
    int    ant_mode           = ANT_MODE_LR_L;  /*< AntMode，见上 */
    double yaw_offset         = 0.0;   /*< 天线安装角 (rad) */
    bool   init_yaw_from_gnss = true;  /*< 初始姿态用双天线航向（不再固定 yaw0=0） */

    /* 模拟 GNSS 中断区间：[t0, t1]，单位 s（相对首帧）。t1 <= t0 表示不启用。
     * 中断期间 GNSS 只作为参考真值记录，不进入因子，用于考核航位推算。 */
    double gnss_gap_t0       = 0.0;
    double gnss_gap_t1       = 0.0;

    /* 预积分残差是否按协方差白化。
     * 默认关：P 里没有零偏/初始状态的不确定度，白化后 IMU 因子会比 GNSS 强
     * 好几个量级，估计会变得"不听 GNSS"，LM 迭代次数也会涨 3~4 倍。
     * 先把零偏初值与过程噪声标定对了再打开（--whiten=1）。 */
    bool   whiten_preint     = false;
    bool   fix_first_pose    = true;

    /* 预积分残差是否带上零偏一阶雅可比（∂Δp/∂b、∂Δv/∂b、∂Δq/∂bg）。
     * 打开后零偏可估、GNSS 拟合更紧；代价是滑窗问题变"硬"，实测迭代次数
     * 3875 -> 22429、求解 1.45 s -> 7.72 s，且部分窗口不收敛。
     * 零偏先验标定好之后再打开更有意义。 */
    bool   bias_jac          = false;

   /* ---- 里程计 ---------------------------------------------------- */
    bool   use_odometer   = false;    /*< 是否启用里程计因子 */
    double odo_vel_scale  = 1.0;      /*< 整体缩放 */
    double odo_vel_std    = 0.05;     /*< 里程速度标准差 (m/s) */
    Eigen::Vector3d  odo_abv{0.0, 0.0, 0.0};    /*< 安装角 [pitch, 0, yaw]，rad */
    Eigen::Vector3d  odo_lvOD{0.0, 0.0, 0.0};   /*< 杆臂（b 系，m） */
    double odo_sodo_init  = 0.0;      /*< 刻度因子初值 */

    /* 每帧对比 CSV 输出（空则不输出） */
    std::string compare_csv_path;

    /* ---- 标定 ---- */
    bool       calib_mode{false};  /*< true: 标定模式; false: 导航模式 */
    CalibState calib_init{};       /*< 标定参数初值（标定/导航均可设） */

    bool        use_visual   = false;
    double      visual_sigma = 1.5 / 460.0;
    double      visual_huber = 1.0;
    std::string cam_dir;                  /* 图像目录（离线预处理用） */
    CameraModel cam_model{};              /* 相机内参 */

    bool   use_gnss_raw      {false};
    bool   use_gnss_dd       {false};
    bool   fix_ambiguity     {false};
    int    max_freq          {2};
    bool   use_wl_nl         {false};
    bool   use_if_combination{false};
    double wl_sigma          {0.05};
    double nl_sigma          {0.01};
    double if_sigma          {0.02};
    Eigen::Vector3d  gnss_ant_lever    {0.0, 0.0, 0.0};
    Eigen::Vector3d  base_pos_ecef     {0.0, 0.0, 0.0};
    std::string rinex_obs_path;
    std::string rinex_nav_path;
    std::string rinex_base_path;
};

struct RunnerStats {
    int    n_frames       = 0;
    int    n_kf           = 0;      /*< 窗口残留关键帧 */
    int    n_kf_total     = 0;      /*< 累计处理的关键帧 */
    int    n_prior_added  = 0;      /*< 边缘化先验成功加入问题的次数 */
    int    n_gnss_used    = 0;
    int    n_static_frames = 0;
    double total_time     = 0.0;
    double final_cost     = 0.0;

    /* ---- GNSS 对比统计（位置误差，单位 m） ---- */
    int    n_cmp_init = 0;   /* 参与"初值 vs GNSS"的帧数 */
    int    n_cmp_opt  = 0;   /* 参与"优化后 vs GNSS"的帧数 */

    double init_max = 0.0, init_mean = 0.0, init_rms = 0.0, init_std = 0.0;
    double opt_max  = 0.0, opt_mean  = 0.0, opt_rms  = 0.0, opt_std  = 0.0;

    /* 分位数：P50 / P90 / P95 / P99 / P100 */
    double init_pct[5] = {0, 0, 0, 0, 0};
    double opt_pct [5] = {0, 0, 0, 0, 0};

    double improve_ratio = 0.0;   /* init_rms / opt_rms */

    /* ---- 估计航向 vs 双天线航向（折算到内部 yaw 后）的一致性 ---- */
    int    n_yaw_cmp     = 0;
    double yaw_diff_mean = 0.0;   /*< 均值 (deg)：系统性偏差，可用来定天线安装方式 */
    double yaw_diff_rms  = 0.0;   /*< 均方根 (deg) */

    /* ---- GNSS 中断期间的航位推算误差（模拟中断时才有意义） ---- */
    int    n_gap_frames  = 0;
    double gap_max_err   = 0.0;   /*< 中断期间 |推算 - GNSS| 最大值 (m) */
    double gap_end_err   = 0.0;   /*< 中断结束时刻的误差 (m) */

    /* ---- 计算量统计 ---- */
    double t_preint      = 0.0;   /*< 预积分（含协方差传播）累计耗时 (s) */
    double t_solve       = 0.0;   /*< Ceres 求解累计耗时 (s) */
    int    n_solve       = 0;
    int    n_solve_ok    = 0;
    int    n_solve_iter  = 0;

    CalibState calib_final{};   /*< 优化结束时的标定参数 */
    int    n_odo_used     = 0;      /*< 进入因子的里程计观测数 */

    std::vector<FrameCompare> per_kf;   /* ← 新增 */
    std::map<double, size_t>  per_kf_time_to_idx;            /* ← 新增：时间→下标 */
};

/* ============================================================================
 * 误差累积器：把每帧的误差按时间收集，最后统一计算统计量
 * ==========================================================================*/
struct ErrorAccumulator {
    std::vector<double> err;
    std::vector<double> time;

    void add(double t, double e) {
        time.push_back(t);
        err.push_back(e);
    }

    int size() const { return static_cast<int>(err.size()); }

    void fill(double &max_v, double &mean_v, double &rms_v,
              double &std_v, double pct[5]) const {
        if (err.empty()) return;

        max_v  = 0.0;
        mean_v = 0.0;
        rms_v  = 0.0;
        for (double e : err) {
            max_v  = std::max(max_v, e);
            mean_v += e;
            rms_v  += e * e;
        }
        const int n = static_cast<int>(err.size());
        mean_v /= n;
        rms_v   = std::sqrt(rms_v / n);

        double var = 0.0;
        for (double e : err) var += (e - mean_v) * (e - mean_v);
        std_v = std::sqrt(var / n);

        std::vector<double> sorted = err;
        std::sort(sorted.begin(), sorted.end());
        pct[0] = sorted[std::min(n - 1, n / 2)];
        pct[1] = sorted[std::min(n - 1, static_cast<int>(n * 0.90))];
        pct[2] = sorted[std::min(n - 1, static_cast<int>(n * 0.95))];
        pct[3] = sorted[std::min(n - 1, static_cast<int>(n * 0.99))];
        pct[4] = sorted.back();
    }
};

/* ============================================================================
 * 滑窗图优化管理器
 * ==========================================================================*/
/* 先验里每个参数块的**身份**。不能存裸指针：std::vector<Frame> 会 reallocate，
 * 滑窗也会 erase，指针会失效；每次使用前按身份重新解析。 */
enum class PriorRefKind { CALIB, FRAME_POSE, FRAME_MIX, AMB };

class GraphOptimizer {
public:
    /* 配置 -----------------------------------------------------------------*/
    struct Options {
        int    max_keyframes     = 10;   /*< 滑窗保留关键帧上限 */
        int    max_iterations    = 50;
        bool   fix_first_pose    = true; /*< gauge fix：固定第一帧 pose */
        double static_vel_sigma  = 0.01; /*< 静止零速因子标准差 (m/s) */
        bool   whiten_preint     = false;/*< 预积分残差按协方差白化（默认关，见 RunnerOptions） */
        bool   bias_jac          = false;/*< 残差里用零偏一阶雅可比（默认关，见 RunnerOptions） */
        int    min_frames_solve  = 2;    /*< 达到该帧数即开始求解（消除冷启动纯外推） */
        /* 标定模式：true 时把标定参数块作为自由变量参与优化；
         * false（导航模式）时固定标定参数块。 */
        bool   calib_mode = false;
        Eigen::Vector3d  odo_abv{0.0, 0.0, 0.0};    /*< 安装角 [pitch, 0, yaw]，rad */
        Eigen::Vector3d  odo_lvOD{0.0, 0.0, 0.0};   /*< 杆臂（b 系，m） */

        bool   use_visual    = false;
        double visual_sigma  = 1.5 / 460.0;
        double visual_huber  = 1.0;
        double inv_depth_min = 1.0 / 500.0;
        double inv_depth_max = 1.0 / 0.5;

        bool   use_gnss_pseudorange{false};
        bool   use_gnss_doppler    {false};
        bool   use_gnss_dd         {false};
        bool   fix_ambiguity       {false};
        int    max_freq            {2};
        bool   use_wl_nl           {false};
        bool   use_if_combination  {false};
        double wl_sigma            {0.05};
        double nl_sigma            {0.01};
        double if_sigma            {0.02};
        double clk_rw_sigma        {1.0e-3};
        double clk_drift_rw_sigma  {1.0e-4};
        Eigen::Vector3d  gnss_ant_lever       {0.0, 0.0, 0.0};
    };

    GraphOptimizer() = default;
    explicit GraphOptimizer(const Options &opt) : opt_(opt) {}

    const Options &options(void) const { return opt_; }
    Options       &options(void) { return opt_; }

    /* 主入口 ---------------------------------------------------------------
     * 传入滑窗内所有关键帧，输出优化结果（原地更新 frames 的参数块）。
     * 若滑窗超限，会自动边缘化最旧帧并把先验保留在内部。 */
    bool optimize(std::vector<Frame> &frames, const Eigen::Vector3d &g_n);

    /* 访问器 */
    const Eigen::MatrixXd  &J_prior(void) const { return J_prior_; }
    const Eigen::VectorXd  &r_prior(void) const { return r_prior_; }
    bool  hasPrior(void) const { return prior_ && prior_->J.rows() > 0; }
    void  clearPrior(void) { prior_.reset(); J_prior_.resize(0, 0); r_prior_.resize(0); }

    void setCalibState(const CalibState &c) { c.toData(calib_data_.data()); }
    CalibState getCalibState() const { CalibState c; c.fromData(calib_data_.data()); return c; }

    std::map<int, double>&       invDepths()       { return inv_depths_; }
    const std::map<int, double>& invDepths() const { return inv_depths_; }

    void setFrameContext(const Eigen::Matrix3d& R_en, const Eigen::Vector3d& p0_ecef) {
        R_en_m3_ = R_en; p0_ecef_v3_ = p0_ecef;
    }
    void setBasePosEcef(const Eigen::Vector3d& p) { base_pos_v3_ = p; }

    bool extractAmbiguityCovariance(ceres::Problem& problem, Eigen::MatrixXd& Q_NN);
    AmbMap& ambiguities() { return ambiguities_; }

private:
    /* 组装 Ceres 问题：把 frames 的所有因子加进去 */
    void buildProblem(ceres::Problem &problem, std::vector<Frame> &frames,
                      const Eigen::Vector3d &g_n, std::vector<double*> *ordered = nullptr);

    /* 先验块的引用信息 */
    struct PriorBlockRef {
        PriorRefKind kind{PriorRefKind::CALIB};
        double       time{0.0};   /*< FRAME_POSE / FRAME_MIX 用 */
        AmbKey       amb{0};      /*< AMB 用 */
    };
    /* 一次边缘化产出的先验：J/r 是切空间上的信息形式，
     * refs/meta 描述每个保留参数块（身份 + ambient/tangent/线性化点）。 */
    struct PriorSpec {
        Eigen::MatrixXd              J;
        Eigen::VectorXd              r;
        std::vector<PriorBlockRef>   refs;
        std::vector<MargPriorBlock>  meta;
    };

public:
    /* 边缘化最旧帧：只取"触及 frames[0]"的残差块做 Schur 补，
     * 消去 frames[0] 的切空间变量，产出新的先验 prior_。
     * 调用方随后可以安全地 erase 掉 frames[0]。
     * （public：runRealData 是自由函数，需要在出窗前调用它。） */
    bool marginalizeOldestFrame(std::vector<Frame> &frames, const Eigen::Vector3d &g_n);

private:
    /* 把 prior_ 的块身份解析成当前问题的参数块指针；顺序与 prior_->meta 一致 */
    bool resolvePriorPointers(const std::vector<Frame> &frames, std::vector<double*> &out);

public:
    Options opt_;
    Eigen::MatrixXd  J_prior_;   /*< 上一次边缘化的先验信息矩阵（镜像，供上层查看） */
    Eigen::VectorXd  r_prior_;   /*< 上一次边缘化的先验信息向量（镜像） */
    std::shared_ptr<PriorSpec> prior_;   /*< 真正的先验（按块描述） */
    std::array<double, NUM_CALIB> calib_data_{};
    std::map<int, double> inv_depths_;

    Eigen::Matrix3d  R_en_m3_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p0_ecef_v3_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d base_pos_v3_ = Eigen::Vector3d::Zero();
    AmbMap ambiguities_;
    /* 各频率槽最近一次的参考星，用于检测参考星切换 */
    std::map<int, int> amb_last_ref_;
    /* 本窗口实际加入问题的模糊度键，用于清理陈旧键 */
    std::set<AmbKey> amb_used_keys_;
    bool amb_warned_basepos_{false};  /*< 基准站坐标无效只告警一次 */
    bool prior_drop_warned_{false};   /*< 先验块对不上只告警一次 */
    bool amb_warned_nodata_{false};   /*< 无 DD 数据只告警一次 */

public:
    /* 求解统计（供上层评估计算量与收敛性） */
    int    stat_solves{0};       /*< 调用 solve 的次数 */
    int    stat_converged{0};    /*< 正常收敛的次数 */
    int    stat_prior_blocks_added{0}; /*< 实际加入问题的先验因子个数 */
    int    stat_iterations{0};   /*< 迭代次数累计 */
    double stat_time_s{0.0};     /*< 求解累计耗时 (s) */
    std::string last_message;    /*< 最近一次求解的结束信息 */
    
};



/* 双天线原始航向 -> 内部姿态 yaw（rad）：ant_mode 折算 + 天线安装角 */
double gnssYaw2AttYaw(double raw_yaw, int ant_mode, double yaw_offset);
double range(double val, double minVal, double maxVal);
int sign(double val, double eps = EPS);
double atan2Ex(double y, double x);
inline double asinEx(double x) { return std::asin(range(x, -1.0, 1.0)); }
double norm(const double *pd, int n);
double norm1(const double *pd, int n);
double normInf(const double *pd, int n);

/* ============================ 融合 ============================ */
void fusion(double *x1, double *p1, const double *x2, const double *p2, int n = 9,
            double *xf = nullptr, double *pf = nullptr);
void fusion(Eigen::Vector3d &x1, Eigen::Vector3d &p1, const Eigen::Vector3d x2, const Eigen::Vector3d p2);
void fusion(Eigen::Vector3d &x1, Eigen::Vector3d &p1, const Eigen::Vector3d x2, const Eigen::Vector3d p2, Eigen::Vector3d &xf, Eigen::Vector3d &pf);



/**
 * @brief 状态空间流形运算（参照 OB_GINS 的 State::operator+/operator-）
 *
 *   1. 状态 x ∈ M = R^3 × SO(3) × R^3 × R^3 × R^3 (× R^1)
 *   2. 对旋转施加右扰动 q' = q ⊗ Exp(δφ)，扰动位于体轴系 b 系；
 *      这样 IMU 零偏、速度等切空间量的物理解释与预积分残差一致。
 *   3. 提供 plus/minus 一对互逆操作，可用于：
 *        - Ceres 自动微分的参数流形
 *        - 残差函数中的误差计算
 *        - 状态预测/更新
 *   4. 桥接现有 double* 数据布局，兼容 pose/mix 双数组约定。
 */

/**
 * @brief 流形加法  x ⊞ δx
 *   p'  = p  + δp
 *   q'  = q  ⊗ Exp(δφ)         （右扰动）
 *   v'  = v  + δv
 *   bg' = bg + δbg
 *   ba' = ba + δba
 */
State plus(const State &s, const DeltaN &delta);
/**
 * @brief 流形减法  a ⊟ b
 *   δp  = a.p  - b.p
 *   δφ  = Log(b.q⁻¹ ⊗ a.q)     （与 plus 对偶）
 *   δv  = a.v  - b.v
 *   δbg = a.bg - b.bg
 *   δba = a.ba - b.ba
 */
DeltaN minus(const State &a, const State &b);

/** @brief 原地更新 s ← s ⊞ δx */
void applyDelta(State &s, const DeltaN &delta);


State   plus(const State &s, const DeltaNOdo &delta);
DeltaNOdo minusWithOdo(const State &a, const State &b);
void    applyDelta(State &s, const DeltaNOdo &delta);

/* ============================================================
 * 数据桥接（与 alg_lib 的 stateToData / stateFromData 等价）
 * ============================================================ */
void  toData(const State &s, double *pose, double *mix, bool with_odometer = false);
State fromData(const double *pose, const double *mix, bool with_odometer = false);

void stateToData(const State &state, double *pose, double *mix, bool with_odometer = false);
void stateFromData(const double *pose, const double *mix, bool with_odometer, State &state);
/* pose = {pE, pN, pU, qx, qy, qz, qw}（与 stateToData 一致）
 * delta = {dpE, dpN, dpU, dφx, dφy, dφz}
 * 定义在 alg_lib.cpp -------------------------------------------------------- */
void posePlus(const double *pose, const double *delta, double *pose_plus);

/* y_minus_x = y ⊟ x：
 *   δp = y.p - x.p
 *   δφ = Log(x.q⁻¹ ⊗ y.q)
 * 定义在 alg_lib.cpp -------------------------------------------------------- */
void poseMinus(const double *y, const double *x, double *y_minus_x);

/* 累积法方程：H += Jᵀ J, b += Jᵀ r
 * 说明：J 的行数对应残差个数（J.row == r.rc），列数对应变量维数。
 *      H 与 b 若为空（row == 0）将按 J.clm 自动初始化。 */
void accumulate_normal_equations(const Eigen::MatrixXd &J, const Eigen::VectorXd &r,
                                 Eigen::MatrixXd &H, Eigen::VectorXd &b);

/* Schur 补边缘化，返回 false 表示维数非法或数值分解失败 */
bool schur_complement(const Eigen::MatrixXd &H, const Eigen::VectorXd &b, int num_marginalized,
                      Eigen::MatrixXd &J_out, Eigen::VectorXd &r_out);

std::map<double, std::vector<VisualObs>>
preprocessImages(const std::string& cam_dir,  const CameraModel& cam,
                 double visual_sigma,
                 int    max_feat = 150,
                 int    min_dist = 20);



struct DataSensor281_t {
    double t;
    double wm[3];
    double vm[3];
    double dS;
    double posgps[3];
    double flagGNSS;
    double vngps[3];
    double satnum;
    double baseline;
    double yaw;      /*< 双天线航向：rad，0~2π 回绕 */
    double yawrms;   /*< 航向标准差：deg（>5 视为不可用，179.99 之类为无效标志） */
    double pos610[3];
    double att610[3];
    double vn610[3];
    double hoop;
    double posstd[3];
};
static_assert(sizeof(DataSensor281_t) == 32 * sizeof(double), "DataSensor281_t 必须严格 256 字节");

/* 读取整个 bin 文件；返回成功读到的帧数 -------------------------------------- */
size_t readSensorFile(const std::string &path, std::vector<DataSensor281_t> &out, size_t max_frames = 0);

/* 有效性判据（与 main.cpp 注释一致）------------------------------------------ */
inline bool isValidGnss(const DataSensor281_t &s) {
    /* flagGNSS 非零且 posgps 非全零 */
    if (s.flagGNSS == 0.0) return false;
    if (s.posgps[0] == 0.0 && s.posgps[1] == 0.0 && s.posgps[2] == 0.0) return false;
    if (s.posgps[0] < -PI || s.posgps[0] > PI) return false;   /* 纬度范围 */
    return true;
}

RunnerStats runRealData(const std::string &bin_path, const RunnerOptions &opt,  const std::string &out_nav_path = "");
RunnerStats runRealData(const std::vector<DataSensor281_t> &raw,  const RunnerOptions &opt, const std::string &out_nav_path = "");


/* ============================================================================
 * 以下为原 src/amb_ils.h 的内容（模糊度整数最小二乘），已并入本头文件
 * ==========================================================================*/
/* ============================================================================
 * 模糊度整数最小二乘（ILS / LAMBDA 风格）—— 原 amb_ils.h，已并入本文件
 *
 * 给定浮点模糊度 a 与其协方差 Q，求
 *      â = argmin_{z ∈ Z^n} (z − a)ᵀ Q⁻¹ (z − a)
 * 并给出 ratio 检验所需的次优二次型。
 *
 * 做法（Teunissen 的 LAMBDA 路线）：
 *   1. 整数 Gauss 变换 + 按条件方差重排，把 Q 去相关 → Qz = Tᵀ Q T（T 整数、|det T| = 1）
 *      去相关只影响**搜索效率**，不影响 ILS 的解（整数幺模变换是格上的一一映射）。
 *   2. 在 Qz 的 LDLᵀ 条件分解上做椭球枚举，用引导解（条件取整）先给出上界。
 *   3. 变换回原空间：a_fixed = T⁻ᵀ z_fixed。
 *
 * 正确性由 test_ils.cpp 用**暴力枚举**在小规模问题上逐例对照（这是唯一的保证方式）。
 * 若枚举节点数超过上限，返回 ok = false（保守：宁可判"未固定"，不给可能错的固定解）。
 * ==========================================================================*/


struct IlsResult {
    bool            ok{false};        /*< 枚举是否正常完成（false 表示不可信） */
    int             n{0};
    Eigen::VectorXd a_float;
    Eigen::VectorXd a_fixed;          /*< ILS 最优整数解（原空间） */
    Eigen::VectorXd a_boot;           /*< 引导解（条件取整），用于对照 */
    double          sq_best{0.0};     /*< 最优二次型 */
    double          sq_second{0.0};   /*< 次优二次型 */
    double          ratio{0.0};       /*< sq_second / sq_best */
    double          ratio_boot{0.0};  /*< 次优(相对引导解) / 最优，用于对照 */
    double          p_boot{0.0};      /*< Bootstrapped 成功率 */
    long            nodes{0};         /*< 枚举节点数 */
    int             n_decorr{0};      /*< 去相关的 Gauss 变换次数 */
    double          cond_before{0.0}; /*< 去相关前 Q 的条件数（对角比） */
    double          cond_after{0.0};  /*< 去相关后 */
    /* 条件标准差（原空间序）：来自原始 Q 的 LDLᵀ 的 D 开方。
     * 用来判断"哪些模糊度最有把握"，是部分固定的排序依据。 */
    Eigen::VectorXd cond_sigma;
    /* 诊断用：整数变换矩阵与它的逆（Ti = T⁻ᵀ），满足 Ti·Tᵀ = I */
    Eigen::MatrixXd T_mat;
    Eigen::MatrixXd Ti_mat;
};

/* ratio 门限：ratio > ratio_thresh 判为可固定（常用 2~3） */
bool ilsResolve(const Eigen::VectorXd &a, const Eigen::MatrixXd &Q,
                IlsResult *out, double ratio_thresh = 3.0, long max_nodes = 4000000L);

/* ---- 部分固定（partial AR） ----
 * 整组 ratio 不够时，按条件标准差从小到大保留最有把握的子集，
 * 逐步缩小固定维数直到 ratio 超门限，返回**能通过的最大子集**。 */
struct IlsPartial {
    bool            ok{false};
    std::vector<int> fixed_idx;      /*< 被固定的模糊度下标（原空间，升序） */
    Eigen::VectorXd a_fixed;         /*< 全长：被固定的取整数，其余保持浮点值 */
    IlsResult       last;            /*< 最后一次成功固定的子集 ILS 结果 */
    int             n_tried{0};      /*< 尝试过的子集个数 */
    int             n_full{0};       /*< 模糊度总数 */
    Eigen::VectorXd cond_sigma;      /*< 条件标准差（原空间序） */
};

bool ilsPartialFix(const Eigen::VectorXd &a, const Eigen::MatrixXd &Q,
                   double ratio_thresh, int min_fix, long max_nodes,
                   IlsPartial *out);

/* 暴力参照实现：在 round(a) ± range 内枚举全部整数格点（只用于小规模测试） */
bool ilsResolveBrute(const Eigen::VectorXd &a, const Eigen::MatrixXd &Q,
                     int range, IlsResult *out);

/* ============================================================================
 * 以下为原 src/gnss_dd.h 的内容（RINEX → 双差 DDObs 生产者），已并入本头文件
 * ==========================================================================*/
/* ============================================================================
 * 从 RINEX 观测/星历生成 Frame::dd_obs —— 原 gnss_dd.h，已并入本文件
 *
 * 背景：工程里原来有一套 GNSS 双差紧组合的残差（GnssDD*Residual）与模糊度管理，
 * 但**没有任何地方生产 dd_obs**，`Frame::has_gnss_raw` 永远是 false，
 * 整条紧组合路径是死代码。这里补上生产者：读 RINEX 3 观测 + 广播星历，
 * 按残差期望的约定生成 DDObs。
 *
 * 符号约定（与 GnssDDPseudorangeFreqResidual / GnssDDCarrierFreqResidual 严格一致）：
 *     dd_X[f] = (X_rover^i − X_rover^k) − (X_base^i − X_base^k)
 *   i = DDObs::sat（非参考星），k = DDObs::ref_sat（参考星）
 *   dd_P 单位 m；dd_L 单位**周**（残差里再乘 λ）；dd_D 单位 Hz
 *   sat_i = DDObs::sat 的星历状态，sat_k = DDObs::ref_sat 的星历状态
 *
 * 参考星**按系统分别选取**（跨系统的双差含系统间偏差，不能混）。
 *
 * 时间：统一用「自 GPS 纪元 1980-01-06 00:00:00 起的秒」，两个文件同为 GPS 时。
 * ==========================================================================*/



/* ============================ 时间 ============================ */
/* 公历 → 自 1980-01-06 00:00:00 起的秒 */
double civilToGpsSec(int year, int month, int day, int hour, int minute, double sec);
/* 反向，便于打印 */
void gpsSecToCivil(double t_gps, int *year, int *month, int *day,
                   int *hour, int *minute, double *sec);

/* ============================ 观测文件 ============================ */
/* 一次"要读的信号"请求：系统号 + 观测类型码，例如 {'G', "L1C"} */
struct ObsRequest {
    char        sys{'G'};
    std::string type;       /*< 观测类型码，如 C1C / L1C / C2W / L2W / D1C */
};

/* 单个卫星在某一历元的观测（按请求顺序排列） */
struct RinexSatObs {
    char sys{0};
    int  prn{0};
    std::array<double, 12> val{};
    std::array<int, 12>    lli{};   /*< 失锁指示（RINEX 3 的 LLI 位） */
    std::array<int, 12>    snr{};
    std::array<bool, 12>   has{};

    double v(int k) const   { return val[static_cast<size_t>(k)]; }
    bool   ok(int k) const  { return has[static_cast<size_t>(k)]; }
};

struct RinexEpoch {
    double t{0.0};               /*< 自 GPS 纪元的秒 */
    int    flag{0};              /*< 历元标志：0 正常，1 本历元前有失锁 */
    std::vector<RinexSatObs> sats;
};

struct RinexObsHeader {
    double          version{0.0};
    double          interval{0.0};
    Eigen::Vector3d approx_pos{Eigen::Vector3d::Zero()};  /*< ECEF，可能为 0 */
    bool            approx_pos_valid{false};
    int             n_epochs{0};
    double          t_first{0.0}, t_last{0.0};
    std::map<char, std::vector<std::string>> obs_types;   /*< 每系统的类型表 */
    std::string     marker;
};

/* 只读请求里列出的信号，避免把 112 MB 文件全塞进内存。
 * reqs 的顺序决定了 RinexSatObs::val/has 的下标。 */
bool readRinex3Obs(const std::string &path,
                   const std::vector<ObsRequest> &reqs,
                   std::vector<RinexEpoch> &epochs,
                   RinexObsHeader *hdr = nullptr);

/* ============================ 广播星历 ============================ */
struct KeplerEph {
    char   sys{'G'};
    int    prn{0};
    int    week{0};
    double toe{0.0}, toc{0.0};
    double af0{0.0}, af1{0.0}, af2{0.0};
    double M0{0.0}, delta_n{0.0}, e{0.0}, sqrt_a{0.0};
    double OMEGA0{0.0}, i0{0.0}, omega{0.0}, OMEGA_DOT{0.0}, idot{0.0};
    double Cuc{0.0}, Cus{0.0}, Crc{0.0}, Crs{0.0}, Cic{0.0}, Cis{0.0};
    double tgd{0.0};
    double accuracy{0.0}, health{0.0};
    bool   valid{false};
};

/* RINEX 3 混合导航文件（.22p / .22n / .22c 都能读）；
 * 解析 GPS/伽利略/QZSS/BDS 的 Kepler 型记录；GLONASS 跳过并计数。 */
bool readRinex3Nav(const std::string &path,
                   std::map<char, std::map<int, std::vector<KeplerEph>>> &out,
                   int *n_glonass_skipped = nullptr);

/* 由广播星历算信号发射时刻的卫星 ECEF 位置与钟差（含传播时延与地球自转改正）。
 *   t_rx  : 接收时刻（GPS 秒）
 *   clk   : 输出卫星钟差（秒，含相对论改正），可为 nullptr */
bool satPosFromEph(const KeplerEph &eph, double t_tx,
                   Eigen::Vector3d &pos, double *clk = nullptr);

/* ============================ 双差生产 ============================ */
/* 一个频率槽上的三种信号码。空串表示该测站/该槽没有这种观测。 */
struct DdSignal {
    char        sys{'G'};
    std::string code_P;   /*< 伪距，如 C1C */
    std::string code_L;   /*< 载波相位，如 L1C */
    std::string code_D;   /*< 多普勒，如 D1C（可空） */
};

struct DdBuildConfig {
    Eigen::Vector3d  base_ecef{Eigen::Vector3d::Zero()};    /*< 基准站 ECEF（必须显式给） */
    Eigen::Vector3d  rover_approx{Eigen::Vector3d::Zero()}; /*< 流动站近似 ECEF（选参考星/仰角用） */
    std::vector<DdSignal> slots;       /*< 依次对应 DDObs 的 slot 0,1,2…；最多 MAX_FREQ 个 */
    double elev_mask_deg{10.0};
    double max_base_dt{0.05};          /*< 流动站/基准站历元最大允许时差（s） */
    int    min_sats{4};                /*< 少于该卫星数不出双差 */
    bool   use_lli_slip{true};         /*< LLI 置位时标记周跳 */
    /* MW 周跳门限（周）。DD MW 的单历元噪声约 0.35 周（λ_WL=0.862 m 对
     * 0.4 m 的码噪声），因此门限取 1.5 周 ≈ 3σ，并改用"与弧段均值的偏差"
     * 而非"与上一历元的差"。早先取 0.75 周 + 逐历元差分，实测 11007 个 DD
     * 误报 17184 次周跳，把弧段切碎、模糊度从 18 涨到 30。 */
    double mw_slip_thresh{2.5};        /*< MW 周跳门限（周）；<=0 关闭 */
    int    mw_min_samples{30};         /*< 滑动窗口至少要这么多历元才判跳变 */
    int    mw_window{50};              /*< 滑动窗口长度（历元）。
                                        * 必须用**滑窗**而不是累积均值：MW 随卫星移动缓慢漂移，
                                        * 累积均值跟不上，偏差迟早越限 → 周期性假周跳。 */
    /* 稳定参考星：按"整个窗口内可用历元数最多"选一颗并在窗口内固定使用。
     * 这能让模糊度键（含 ref_sat）保持稳定，维数不再随参考星切换而膨胀
     * （实测 120 历元窗口：逐历元选星 40 个模糊度 → 稳定选星 18 个）。
     * 某历元若选定参考星不可用，则跳过该系统（换参考星会引入新参数）。 */
    bool   stable_reference{true};
};

/* 一个历元的双差结果 + 诊断 */
struct DdEpochResult {
    double t{0.0};
    int    ref_prn{0};                 /*< 第一个系统的参考星（诊断用） */
    std::vector<DDObs> dd;
    /* 与 dd 一一对应：该星在每个槽上是否检测到周跳（LLI 或 MW） */
    std::vector<std::array<bool, MAX_FREQ>> slip;
    int    n_common{0};                /*< 双站共同可见且仰角合格的卫星数 */
    int    n_used{0};                  /*< 实际形成双差的卫星数 */
    int    n_sys{0};                   /*< 参与的系统数 */
};

/* 双差的 MW（Melbourne-Wübbena）宽巷模糊度，单位周。
 * 需要 f1/f2 两个槽的 P 与 L 都有效；否则返回 false。 */
bool ddMelbourneWubbena(const DDObs &dd, int f1, int f2, double *n_wl);

/* 组双差的统计（可选输出，便于诊断参考星策略的效果） */
struct DdBuildStats {
    int n_epochs_rover{0};        /*< 输入的流动站历元数 */
    int n_no_base{0};             /*< 找不到对齐的基准站历元 */
    int n_too_few{0};             /*< 共同可见卫星不足 min_sats */
    int n_ref_missing{0};         /*< 选定参考星在该历元不可用而跳过（按系统计） */
    int n_epochs_out{0};          /*< 最终产出的历元数 */
    int n_slip{0};                /*< 判出的周跳次数（按 (DD,槽) 计） */
    std::map<char, int> chosen_ref;   /*< 系统 → 全窗口参考星 PRN */
};

/* 把 rover/base 的观测配对（按时间）、算卫星位置、组双差。 */
bool buildDdEpochs(const std::vector<RinexEpoch> &rover,
                   const std::vector<RinexEpoch> &base,
                   const std::map<char, std::map<int, std::vector<KeplerEph>>> &eph,
                   const DdBuildConfig &cfg,
                   std::vector<DdEpochResult> &out,
                   DdBuildStats *stats = nullptr);

/* 由 slots 生成读取用的请求表（顺序固定：每个 slot 依次 P/L/D），
 * 便于调用方用同一份 reqs 去读流动站与基准站。 */
std::vector<ObsRequest> buildObsRequests(const std::vector<DdSignal> &slots,
                                         std::vector<std::array<int, 3>> *slot_index);

/* 把 DdEpochResult 按时间装进 Frame（设置 dd_obs / has_gnss_raw）。
 *   t_offset : gps_sec = frame_time + t_offset
 *   tol      : 时间匹配容差（s）
 * 返回成功挂上的帧数。 */
int attachDdToFrames(std::vector<Frame> &frames,
                     const std::vector<DdEpochResult> &dd_epochs,
                     double t_offset, double tol = 0.6);

#endif

