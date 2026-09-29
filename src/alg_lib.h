/**
 * @file alg_lib.h
 * @brief 算法库声明（数学 / 地球模型 / IMU 预积分 / 图优化）
 *
 * 【不要直接引用本文件】对外只需要 #include "app_interface.h"，
 * 由它在末尾统一引入本文件；两个头都有 guard，顺序无所谓。
 *
 * 分节（与 alg_lib.cpp 的实现顺序一一对应）：
 *   0. 基础常量 ................ app_interface.h
 *   1. 基础类型 ................ vect3 / mat3 / quat / vect / mat
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

/* 参数块维数 ---------------------------------------------------------------------*/
constexpr int NUM_POSE      = 7;  /*< p(3) + q(4) */
constexpr int NUM_MIX       = 9;  /*< v(3) + bg(3) + ba(3) */
constexpr int NUM_MIX_ODO   = 10; /*< 再加上里程计比例因子误差 sodo */
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

class mat3;
class quat;
class mat;   
class vect; 
class vect3;
class earth;
class Preintegration;

/* ---------- 常用常量（定义在 ipos_fixed.cpp） ---------- */
extern vect3 O31;   /*< 零矢量 */
extern vect3 One31; /*< 全 1 矢量 */
extern mat3 I33;    /*< 单位阵 */
extern quat qI;     /*< 单位四元数 */

mat3 Rot(double angle, char axis);
mat3 rcijk(const mat3 &m, int ijk);
void normlize(quat *q);

class vect3 {
public:
    double i, j, k;

    vect3(void) : i(0.0), j(0.0), k(0.0) {}
    vect3(double xyz) : i(xyz), j(xyz), k(xyz) {}
    vect3(double xx, double yy, double zz) : i(xx), j(yy), k(zz) {}
    explicit vect3(const double *pdata) : i(pdata[0]), j(pdata[1]), k(pdata[2]) {}
    explicit vect3(const float *pdata) : i(pdata[0]), j(pdata[1]), k(pdata[2]) {}
    vect3(const Eigen::Vector3d &v) : i(v(0)), j(v(1)), k(v(2)) {}

    /* ---------- Eigen 互操作 ---------- */
    Eigen::Map<Eigen::Vector3d> e(void) { return Eigen::Map<Eigen::Vector3d>(&i); }
    Eigen::Map<const Eigen::Vector3d> e(void) const { return Eigen::Map<const Eigen::Vector3d>(&i); }
    Eigen::Vector3d toEigen(void) const { return Eigen::Vector3d(i, j, k); }

    /* ---------- 与原版一致的成员查询 ---------- */
    int IsZeros(const vect3 &v, double eps = EPS) const;
    bool IsZeroXY(const vect3 &v, double eps = EPS) const;
    bool IsNaN(const vect3 &v) const;

    /* ---------- 运算 ---------- */
    vect3 &operator=(double f);
    vect3 &operator=(const double *pf);

    vect3 operator+(const vect3 &v) const;
    vect3 operator-(const vect3 &v) const;
    vect3 operator*(const vect3 &v) const; /*< 叉乘 */
    vect3 operator*(const mat &m) const;   /*< 行向量 × 矩阵 */
    vect3 operator*(double f) const;
    vect3 operator*(const mat3 &m) const; /*< 行向量 × 矩阵 */
    vect3 operator/(double f) const;
    vect3 operator/(const vect3 &v) const;

    vect3 &operator+=(const vect3 &v);
    vect3 &operator-=(const vect3 &v);
    vect3 &operator*=(double f);
    vect3 &operator/=(double f);
    vect3 &operator/=(const vect3 &v);

    friend vect3 operator*(double f, const vect3 &v);
    friend vect3 operator-(const vect3 &v);
};

/* ============================================================================
 * mat3
 * ==========================================================================*/
class mat3 {
public:
    double e00, e01, e02, e10, e11, e12, e20, e21, e22;

    mat3(void) { set(0, 0, 0, 0, 0, 0, 0, 0, 0); }
    mat3(double xyz) { set(xyz, xyz, xyz, xyz, xyz, xyz, xyz, xyz, xyz); }
    explicit mat3(const double *pxyz);
    explicit mat3(const float *pxyz);
    mat3(double xx, double yy, double zz) { set(xx, 0, 0, 0, yy, 0, 0, 0, zz); }
    mat3(double xx, double xy, double xz, double yx, double yy, double yz, double zx, double zy,
         double zz) {
        set(xx, xy, xz, yx, yy, yz, zx, zy, zz);
    }
    mat3(const vect3 &v0, const vect3 &v1, const vect3 &v2, bool isrow = 1);
    mat3(const Eigen::Matrix3d &m);

    void set(double xx, double xy, double xz, double yx, double yy, double yz, double zx, double zy,
             double zz) {
        e00 = xx; e01 = xy; e02 = xz;
        e10 = yx; e11 = yy; e12 = yz;
        e20 = zx; e21 = zy; e22 = zz;
    }

    double &operator()(int i, int j = -1);
    double operator()(int i, int j = -1) const;

    /* ---------- Eigen 互操作 ---------- */
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> e(void) {
        return Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(&e00);
    }
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> e(void) const {
        return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(&e00);
    }
    Eigen::Matrix3d toEigen(void) const { return e(); }

    /* ---------- 运算 ---------- */
    mat3 operator+(const mat3 &m) const;
    mat3 operator-(const mat3 &m) const;
    mat3 operator*(const mat3 &m) const;
    mat3 operator*(double f) const;
    vect3 operator*(const vect3 &v) const;
    mat3 &operator+=(const mat3 &m);
    mat3 operator+(const vect3 &v) const; /*< 矢量加到对角线 */
    mat3 &operator+=(const vect3 &v);

    void SetRow(int i, const vect3 &v);
    void SetClm(int i, const vect3 &v);
    vect3 GetRow(int i) const;
    vect3 GetClm(int i) const;

    mat3 Rot(double angle, char axis) { return ::Rot(angle, axis); }
    mat3 rcijk(const mat3 &m, int ijk) { return ::rcijk(m, ijk); }

    friend mat3 operator-(const mat3 &m);
    friend mat3 operator~(const mat3 &m);
    friend mat3 operator*(double f, const mat3 &m);
};

/* ============================================================================
 * quat  (q0 = w, q1..q3 = x,y,z)
 * ==========================================================================*/
class quat {
public:
    double q0, q1, q2, q3;

    quat(void) : q0(0.0), q1(0.0), q2(0.0), q3(0.0) {}
    quat(double qq0, double qq1 = 0.0, double qq2 = 0.0, double qq3 = 0.0)
        : q0(qq0), q1(qq1), q2(qq2), q3(qq3) {}
    quat(double qq0, const vect3 &qqv) : q0(qq0), q1(qqv.i), q2(qqv.j), q3(qqv.k) {}
    quat(const double *pdata) : q0(pdata[0]), q1(pdata[1]), q2(pdata[2]), q3(pdata[3]) {}
    quat(const Eigen::Quaterniond &q) : q0(q.w()), q1(q.x()), q2(q.y()), q3(q.z()) {}

    /* ---------- Eigen 互操作 ---------- */
    Eigen::Quaterniond toEigen(void) const { return Eigen::Quaterniond(q0, q1, q2, q3); }

    /* ---------- 运算 ---------- */
    quat operator+(const vect3 &phi) const;
    quat operator-(const vect3 &phi) const;
    vect3 operator-(const quat &quat) const; /*< 失准角 */
    quat operator*(const quat &q) const;
    vect3 operator*(const vect3 &v) const; /*< 旋转矢量 */
    quat &operator*=(const quat &q);
    quat &operator-=(const vect3 &phi);

    void SetYaw(double yaw = 0.0);
    void normlize(quat *q) { ::normlize(q); }

    friend quat operator~(const quat &q);
};

/* ============================================================================
 * vect
 * ==========================================================================*/
class vect {
public:
    int row, clm, rc; /*< 与原版一致的形状描述：clm==1 时为列向量，否则为行向量 */
    RowMajorMatrix E; /*< 实际存储（行主序），尺寸为 row×clm */
    double *dd;       /*< == E.data()，供 `dd[i]` 风格的既有代码使用 */

    vect(void);
    vect(int row0, int clm0 = 1);
    vect(int row0, double f);
    vect(int row0, double f, double f1, ...);
    vect(int row0, const double *pf);
    vect(const vect3 &v);
    vect(const vect3 &v1, const vect3 v2);
    vect(const Eigen::VectorXd &v);

    vect(const vect &o);
    vect &operator=(const vect &o);

    /* ---------- Eigen 互操作 ---------- */
    Eigen::Map<const RowMajorMatrix> e(void) const {
        return Eigen::Map<const RowMajorMatrix>(dd, row, clm);
    }
    Eigen::Map<RowMajorMatrix> e(void) { return Eigen::Map<RowMajorMatrix>(dd, row, clm); }
    Eigen::VectorXd toEigen(void) const { return Eigen::Map<const Eigen::VectorXd>(dd, rc); }
    int size(void) const { return rc; }

    /* ---------- 与原版一致的成员函数 ---------- */
    void Set(double f[], int size);
    void Set2(double f[], int size);
    void SetVect3(int i, const vect3 &v);
    void Set2Vect3(int i, const vect3 &v);
    void SetBit(unsigned int bit, double f);
    void SetBit(unsigned int bit, const vect3 &v);
    vect3 GetVect3(int i) const;

    vect operator+(const vect &v) const;
    vect operator-(const vect &v) const;
    vect operator*(double f) const;
    vect &operator=(double f);
    vect &operator=(const double *pf);
    vect &operator=(const mat3 &m);
    vect &operator+=(const vect &v);
    vect &operator-=(const vect &v);
    vect &operator*=(double f);
    mat operator*(const vect &v) const; /*< 外积 */
    double &operator()(int r) { return dd[r]; }

    friend vect operator~(const vect &v);

private:
    void resize(int r, int c);
    void bind(void) { dd = E.data(); }
};

/* ============================================================================
 * mat
 * ==========================================================================*/
class mat {
public:
    int row, clm, rc;
    RowMajorMatrix E;
    double *dd;

    mat(void);
    mat(int row0, int clm0);
    mat(int row0, int clm0, double f);
    mat(int row0, int clm0, double f, double f1, ...);
    mat(int row0, int clm0, const double *pf);
    mat(const Eigen::MatrixXd &m);

    mat(const mat &o);
    mat &operator=(const mat &o);

    /* ---------- Eigen 互操作 ---------- */
    Eigen::Map<const RowMajorMatrix> e(void) const {
        return Eigen::Map<const RowMajorMatrix>(dd, row, clm);
    }
    Eigen::Map<RowMajorMatrix> e(void) { return Eigen::Map<RowMajorMatrix>(dd, row, clm); }
    Eigen::MatrixXd toEigen(void) const { return e(); }
    int size(void) const { return rc; }

    /* ---------- 与原版一致的成员函数 ---------- */
    void Clear(void);
    void SetDiag(double f[], int len);
    void SetDiag2(double f[], int len);

    mat operator+(const mat &m) const;
    mat operator-(const mat &m) const;
    mat operator*(double f) const;
    vect operator*(const vect &v) const;
    mat operator*(const mat &m) const;
    mat &operator=(double f);
    mat &operator+=(const mat &m0);
    mat &operator+=(const vect &v);
    mat &operator-=(const mat &m0);
    mat &operator*=(double f);
    mat &operator++();
    double &operator()(int r, int c = -1) { return dd[r * clm + (c < 0 ? r : c)]; }
    /*< 只读访问（const 版本，Eigen 化之后补上的便利接口） */
    double operator()(int r, int c = -1) const { return dd[r * clm + (c < 0 ? r : c)]; }

    void ZeroRow(int i);
    void ZeroClm(int j);
    void SetRow(int i, double f, ...);
    void SetRow(int i, const vect &v);
    void SetClm(int j, double f[], int len);
    void SetClm(int j, const vect &v);
    vect GetRow(int i) const;
    void GetRow(vect &v, int i);
    vect GetClm(int j) const;
    void GetClm(vect &v, int j);

    void SetRowVect3(int i, int j, const vect3 &v);
    void SetRowVect3(int i, int j, const vect3 &v, const vect3 &v1);
    void SetRowVect3(int i, int j, const vect3 &v, const vect3 &v1, const vect3 &v2);
    void SetClmVect3(int i, int j, const vect3 &v);
    void SetClmVect3(int i, int j, const vect3 &v, const vect3 &v1);
    void SetClmVect3(int i, int j, const vect3 &v, const vect3 &v1, const vect3 &v2);
    vect3 GetRowVect3(int i, int j) const;
    vect3 GetClmVect3(int i, int j) const;
    void SetDiagVect3(int i, int j, const vect3 &v);
    vect3 GetDiagVect3(int i, int j = -1) const;
    void SetAskew(int i, int j, const vect3 &v);
    void SetMat3(int i, int j, const mat3 &m);
    void SetMat3(int i, int j, const mat3 &m, const mat3 &m1);
    void SetMat3(int i, int j, const mat3 &m, const mat3 &m1, const mat3 &m2);
    mat3 GetMat3(int i, int j = -1) const;
    void SubAddMat3(int i, int j, const mat3 &m);

    friend mat operator~(const mat &m);

private:
    void resize(int r, int c);
    void bind(void) { dd = E.data(); }
};

/* ============================================================================
 * 辅助：vect3 → Eigen 类型（支持 AutoDiff 的 Jet<T>）
 * ==========================================================================*/
template <typename T>
inline Eigen::Matrix<T, 3, 1> toEig3(const vect3 &v) {
    return Eigen::Matrix<T, 3, 1>(T(v.i), T(v.j), T(v.k));
}
template <typename T>
inline Eigen::Quaternion<T> toEigQ(const quat &q) {
    return Eigen::Quaternion<T>(T(q.q0), T(q.q1), T(q.q2), T(q.q3));
}
/* ============================ 姿态 / 旋转 ============================ */
mat3 a2mat(const vect3 &att);           /*< [pitch,roll,yaw]   -> Cnb  */
quat a2qua(double pitch, double roll, double yaw);
quat a2qua(const vect3 &att);
vect3 m2att(const mat3 &Cnb);           /*< Cnb -> [pitch,roll,yaw]    */
vect3 q2att(const quat &qnb);

mat3 ar2mat(const vect3 &attr);         /*< 反序欧拉角 -> Cnb          */
quat ar2qua(const vect3 &attr);
vect3 m2attr(const mat3 &Cnb);
vect3 q2attr(const quat &qnb);

quat rv2q(const vect3 &rv);             /*< 旋转矢量 -> 四元数         */
mat3 rv2m(const vect3 &rv);
vect3 q2rv(const quat &q);
vect3 m2rv(const mat3 &Cnb);

mat3 q2mat(const quat &qnb);
quat m2qua(const mat3 &Cnb);
mat3 askew(const vect3 &v);

vect3 qq2phi(const quat &qcalcu, const quat &qreal);
quat addmu(const quat &q, const vect3 &mu);
quat UpDown(const quat &q);
vect3 q2att(const quat &qnb);

vect3 sv2att(const vect3 &fb, double yaw0, const vect3 &fn);
vect3 vn2att(const vect3 &vn);
double vn2att(double vel_east, double vel_north);

/* ============================ mat3 线性代数 ============================ */
mat3 Rot(double angle, char axis);
mat3 rcijk(const mat3 &m, int ijk);
double trMMT(const mat3 &m1, const mat3 &m2 = I33);
void symmetry(mat3 &m);
mat3 pow(const mat3 &m, int k);
double trace(const mat3 &m);
double det(const mat3 &m);
mat3 adj(const mat3 &m);
mat3 inv(const mat3 &m);
vect3 diag(const mat3 &m);
mat3 diag(const vect3 &v);
mat3 askew(const mat3 &m, int I);
mat3 dotmul(const mat3 &m1, const mat3 &m2);
mat3 MMT(const mat3 &m1, const mat3 &m2);
double norm(const mat3 &m);
mat3 randn(const mat3 &mu, const double &sigma);

/* ============================ vect 线性代数 ============================ */
vect abs(const vect &v);
double norm(const vect &v);
double norm1(const vect &v);
double normInf(const vect &v);
vect pow(const vect &v, int k);
vect sort(const vect &v);
double dot(const vect &v1, const vect &v2);
vect dotmul(const vect &v1, const vect &v2);
vect randn(const vect &mu, const vect &sigma);

/* ============================ mat 线性代数 ============================ */
void symmetry(mat &m);
double trace(const mat &m);
double norm1(const mat &m);
double normInf(const mat &m);
mat dotmul(const mat &m1, const mat &m2);
vect diag(const mat &m);
mat diag(const vect &v);
mat eye(int n);
mat inv4(const mat &m);
void RowMul(mat &m, const mat &m0, const mat &m1, int r, int fast = 0);
void RowMulT(mat &m, const mat &m0, const mat &m1, int r, int fast = 0);
void DVMDVafa(const vect &V, mat &M, double afa);
mat randn(const mat &mu, const double &sigma);
mat llt_L(const mat &A);
mat llt_sqrtinv(const mat &A);
mat llt_inv(const mat &A);
mat pinv_sym(const mat &A, double eps = 1e-10);
void accumulate_normal_equations(const mat &J, const vect &r, mat &H, vect &b);

/* Schur 补 ----------------------------------------------------------------------
 *   H, b  : 完整法方程（前 num_marginalized 维是被边缘化的状态）
 *   J_out : n_keep × n_keep，满足 J_out^T J_out = H*
 *   r_out : n_keep，满足 J_out^T r_out = b*
 * 返回 false 表示维数非法或分解失败。 -------------------------------------------*/
bool schur_complement(const mat &H, const vect &b, int num_marginalized, mat &J_out, vect &r_out);
/* ============================ 坐标系 ============================ */
mat3 pos2Cen(const vect3 &pos);
vect3 xyz2blh(const vect3 &xyz);
vect3 blh2xyz(const vect3 &blh);
vect3 Vxyz2enu(const vect3 &Vxyz, const vect3 &pos);


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

    vect3 pos;  /*< {lat, lon, h} */
    vect3 vn;   /*< {vE, vN, vU} */
    vect3 wnie; /*< 地球自转在 n 系下的表示 */
    vect3 wnen; /*< n 系相对 e 系的转动（运输率） */
    vect3 wnin; /*< n 系相对 i 系的转动 */
    vect3 gn;   /*< 重力矢量（ENU: {0,0,-g}） */
    vect3 gcc;  /*< 重力 + 科氏/向心修正 */
    vect3 *pgn; /*< 若不为空则用它替换 gn */

    earth(double a = RE, double f = f0_earth);

    void Init(double a = RE, double f = f0_earth);

    /* 由位置和速度刷新椭球/重力/角速度参数 -----------------------------------------*/
    void Update(const vect3 &pos, const vect3 &vn);

    /* 速度 -> 位置增量（{dLat, dLon, dh}） -----------------------------------------*/
    vect3 vn2dpos(const vect3 &vn, float ts = 1.0f) const;
};

class LocalFrame {
public:
    LocalFrame(void) { init(vect3(0.0)); }

    /* 以 blh0 为原点建立局部 NEU 系 ----------------------------------------------*/
    void init(const vect3 &blh0);

    bool isInit(void) const { return init_; }
    const vect3 &blh0(void) const { return blh0_; }

    vect3 blh2neu(const vect3 &blh) const;
    vect3 neu2blh(const vect3 &neu) const;

    /* 参考点处的 earth 模型（wnie / gn / RMh / RNh 都取这里的值） ------------------*/
    const earth &earthModel(void) const { return eth_; }

    /* ENU 下的重力矢量 {0, 0, -g} ------------------------------------------------*/
    const vect3 &gravity(void) const { return gravity_; }

    /* ENU 下的地球自转 {0, wie*cos(lat0), wie*sin(lat0)} ------------------------*/
    const vect3 &wnie(void) const { return eth_.wnie; }

private:
    bool init_{false};
    vect3 blh0_;
    vect3 gravity_;
    earth eth_;
};


class IMU {
public:
    double tk;       /*< 当前时间 */
    float nts;       /*< 本次更新覆盖的时间 */
    float _nts;      /*< 1 / nts */
    mat3 Kg;         /*< 陀螺标定阵 */
    vect3 eb;        /*< 陀螺零偏 */
    mat3 Ka;         /*< 加表标定阵 */
    vect3 db;        /*< 加表零偏 */

    vect3 phim;      /*< 角度增量（已含圆锥补偿） */
    vect3 dvbm;      /*< 速度增量（已含划桨补偿） */
    vect3 wmm;       /*< 角速率测量之和 */
    vect3 vmm;       /*< 比力测量之和 */
    vect3 swmm;      /*< 角速率累加 */
    vect3 svmm;      /*< 比力累加 */
    vect3 wm_1;      /*< 上一采样角增量 */
    vect3 vm_1;      /*< 上一采样速度增量 */

    int nSamples;    /*< 本次更新的子样数（1~5） */
    bool preFirst;   /*< 是否为上电后的第一次更新 */
    bool onePlusPre; /*< 单子样时是否补上上一采样 */
    bool preWb;      /*< 是否已有上一采样 */

    IMU(void);

    /* 复位（保留标定参数），onePlusPre 默认打开 --------------------------------*/
    void Reset(void);

    void SetKga(const mat3 &Kg0, const vect3 eb0, const mat3 &Ka0, const vect3 &db0);

    /* 由等间隔采样增量计算 phim / dvbm
     *   pwm[i] : 第 i 个子样的角增量  (rad)
     *   pvm[i] : 第 i 个子样的速度增量 (m/s)
     *   nSamples: 1~5，ts: 单个子样的时间
     * 返回更新后的时间 tk -----------------------------------------------------------*/
    double Update(const vect3 *pwm, const vect3 *pvm, int nSamples, double ts);
};

struct PreintegrationParam {
    double gyr_arw{0.0};      /*< 角度随机游走 rad/sqrt(s) */
    double acc_vrw{0.0};      /*< 速度随机游走 m/s^1.5 */
    double gyr_bias_std{0.0}; /*< 陀螺零偏标准差 rad/s */
    double acc_bias_std{0.0}; /*< 加表零偏标准差 m/s^2 */
    double corr_time{3600.0}; /*< 零偏一阶马尔可夫相关时间 s */

    /* ---- 里程计（use_odometer = true 时生效）---- */
    vect3 odo_std;             /*< 单次里程增量的白噪声标准差 (m) */
    double odo_srw{0.0};       /*< 里程计比例因子随机游走 (1/sqrt(s)) */
    vect3 abv;                 /*< 安装角 [pitch, 0, yaw]，与 ipos3g 的 odo::ODKappa 一致 */
    vect3 lvOD;                /*< 里程计杆臂（b 系，m） */
};

/* 关键帧状态 ---------------------------------------------------------------------*/
struct State {
    double time{0.0};
    vect3 p;                       /*< 局部 NEU 位置 (m) */
    quat q{1.0, 0.0, 0.0, 0.0};    /*< b -> n */
    vect3 v;                       /*< n 系速度 (m/s) */
    vect3 bg;                      /*< 陀螺零偏 (rad/s) */
    vect3 ba;                      /*< 加表零偏 (m/s^2) */
    double sodo{0.0};              /*< 里程计比例因子误差（无量纲） */
    vect3 s;                       /*< 预积分里程位移（b0 系，m），仅内部使用 */
};

/* 观测 ---------------------------------------------------------------------------*/
struct ImuMeas {
    double time{0.0};
    double dt{0.0};
    vect3 dtheta; /*< 角增量 (rad) */
    vect3 dvel;   /*< 速度增量 (m/s) */
    double odovel{0.0}; /*< 本次采样间隔内的里程增量 (m)，对应 OB_GINS 的 imu.odovel */
};

struct GnssMeas {
    double time{0.0};
    vect3 blh;    /*< {lat, lon, h}，弧度/米 */
    vect3 std;    /*< 位置 {sigma_E, sigma_N, sigma_U}，米（局部 NEU） */
    vect3 vn;     /*< n 系速度 (m/s)，需要时填（ipos3g 的 KF 就用它） */
    vect3 vn_std; /*< 速度标准差 (m/s) */
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
    vect3  xyz_c{0.0, 0.0, 0.0};   /* (X/Z, Y/Z, 1) */
    double sigma{1.5 / 460.0};
};

/* ---- 相机针孔模型 + 径向切向畸变 ---- */
struct CameraModel {
    double fx{0.0}, fy{0.0}, cx{0.0}, cy{0.0};
    double k1{0.0}, k2{0.0}, p1{0.0}, p2{0.0};
    cv::Mat K() const {
        return (cv::Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
    }
    cv::Mat D() const {
        return (cv::Mat_<double>(1,4) << k1,k2,p1,p2);
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
    vect3  lvOD{0.0, 0.0, 0.0};
    double yaw_gnss_offset{0.0};
    /* ---- 相机外参（新增） ---- */
    vect3  t_bc{0.0, 0.0, 0.0};
    quat   q_bc{1.0, 0.0, 0.0, 0.0};

    void toData(double *d) const {
        d[0]=sodo; d[1]=abv_pitch; d[2]=abv_yaw;
        d[3]=lvOD.i; d[4]=lvOD.j; d[5]=lvOD.k;
        d[6]=yaw_gnss_offset;
        d[7]=t_bc.i; d[8]=t_bc.j; d[9]=t_bc.k;
        d[10]=q_bc.q1; d[11]=q_bc.q2; d[12]=q_bc.q3; d[13]=q_bc.q0;
    }
    void fromData(const double *d) {
        sodo=d[0]; abv_pitch=d[1]; abv_yaw=d[2];
        lvOD=vect3(d[3],d[4],d[5]); yaw_gnss_offset=d[6];
        t_bc=vect3(d[7],d[8],d[9]);
        q_bc=quat(d[13],d[10],d[11],d[12]);
    }
};

/* 一个关键帧 ----------------------------------------------------------------------*/
struct Frame {
    double time{0.0};
    /* Ceres 参数块（顺序与 gopt_types.h 一致）。数组固定长度，保证地址稳定 ----*/
    std::array<double, NUM_POSE> pose{};     /*< {pE,pN,pU, qx,qy,qz,qw} */
    std::array<double, NUM_MIX_ODO> mix{};   /*< {v, bg, ba [, sodo]} */

    /* 到下一关键帧的预积分（最后一帧为空） ----------------------------------*/
    std::shared_ptr<Preintegration> pre_to_next;

    /* 该帧上的 GNSS 位置观测（已转到局部 ENU） -------------------------------*/
    bool has_gnss{false};
    vect3 gnss_pos{};
    vect3 gnss_std{1.0, 1.0, 1.0};
    vect3 gnss_std0{1.0, 1.0, 1.0}; /*< 原始 std，重加权时以它为基准（避免累乘） */
    double gnss_time{0.0};
    bool   gnss_in_gap{false};      /*< 该帧落在模拟的 GNSS 中断区间（只做参考，不入因子） */

    /* 该帧上的 GNSS 速度观测 ------------------------------------*/
    bool has_gnss_vel{false};
    vect3 gnss_vn{};
    vect3 gnss_vn_std{1.0, 1.0, 1.0};

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
    vect3  odo_omega_meas{};          /*< 该段陀螺原始测量 (rad/s)，用于杆臂补偿 */
    double odo_std{0.05};             /*< 里程速度标准差 (m/s) */

    std::vector<VisualObs> visual_obs;   /* 新增 */
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

        quat q(x[6], x[3], x[4], x[5]);
        quat dq = rv2q(vect3(delta[3], delta[4], delta[5]));
        quat q_new = q * dq;
        normlize(&q_new);

        x_plus[3] = q_new.q1;
        x_plus[4] = q_new.q2;
        x_plus[5] = q_new.q3;
        x_plus[6] = q_new.q0;
        return true;
    }

    bool PlusJacobian(const double *x, double *jacobian) const override {
        /* PoseManifold 的切空间是 6 维，ambient 是 7 维：
        *   切空间 δ = [δp(3), δφ(3)]
        *   ambient = [p(3), qx, qy, qz, qw]
        * 输出缓冲区是 7×6 的 RowMajor 矩阵。 */
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
        J.setZero();

        /* 位置部分：∂p/∂δp = I_3 */
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

        /* 旋转部分：q' = q ⊗ Exp(δφ)，在 δφ = 0 处 ∂q/∂δφ = 0.5·Q(q)
        * 但 pose 中四元数顺序是 [qx, qy, qz, qw]，需要重排行序： */
        const double qx = x[3], qy = x[4], qz = x[5], qw = x[6];

        /* 与 Ceres 内置 QuaternionManifold 一致：
        *   ∂qx/∂δφ = 0.5·( qw, -qz,  qy)
        *   ∂qy/∂δφ = 0.5·( qz,  qw, -qx)
        *   ∂qz/∂δφ = 0.5·(-qy,  qx,  qw)
        *   ∂qw/∂δφ = 0.5·(-qx, -qy, -qz)  */
        J(3, 3) =  0.5 * qw; J(3, 4) = -0.5 * qz; J(3, 5) =  0.5 * qy;
        J(4, 3) =  0.5 * qz; J(4, 4) =  0.5 * qw; J(4, 5) = -0.5 * qx;
        J(5, 3) = -0.5 * qy; J(5, 4) =  0.5 * qx; J(5, 5) =  0.5 * qw;
        J(6, 3) = -0.5 * qx; J(6, 4) = -0.5 * qy; J(6, 5) = -0.5 * qz;
        return true;
    }
    bool Minus(const double *y, const double *x, double *y_minus_x) const override {
        y_minus_x[0] = y[0] - x[0];
        y_minus_x[1] = y[1] - x[1];
        y_minus_x[2] = y[2] - x[2];

        quat qx(x[6], x[3], x[4], x[5]);
        quat qy(y[6], y[3], y[4], y[5]);
        vect3 dphi = q2rv((~qx) * qy);
        y_minus_x[3] = dphi.i;
        y_minus_x[4] = dphi.j;
        y_minus_x[5] = dphi.k;
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

        /* 旋转部分：pose 中四元数顺序为 [qx, qy, qz, qw] */
        const double qx = x[3], qy = x[4], qz = x[5], qw = x[6];

        Eigen::Matrix<double, 3, 4> Jq;
        Jq << -qy,  qx, -qw,  qz,
              -qz,  qw,  qx, -qy,
              -qw, -qz,  qy,  qx;
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
        quat q(x[13], x[10], x[11], x[12]);   /* (w, x, y, z) */
        quat dq = rv2q(vect3(delta[10], delta[11], delta[12]));
        quat q_new = q * dq;
        normlize(&q_new);
        x_plus[10] = q_new.q1;
        x_plus[11] = q_new.q2;
        x_plus[12] = q_new.q3;
        x_plus[13] = q_new.q0;
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
        quat qy(y[13], y[10], y[11], y[12]);
        quat qx(x[13], x[10], x[11], x[12]);
        vect3 dphi = q2rv((~qx) * qy);
        y_minus_x[10] = dphi.i;
        y_minus_x[11] = dphi.j;
        y_minus_x[12] = dphi.k;
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
    void setBias(const vect3 &bg, const vect3 &ba);

    /* ---- 丢弃 repropagation 用的历史测量 ----
     * 滑窗优化里 pre_to_next 只用来求残差，不会再改零偏重放，
     * 因此冻结时可以丢掉这段历史，避免每帧拷贝 ~200 条 ImuMeas。 */
    void dropHistory(void) { meas_list_.clear(); }

    /* ---- 访问器 ---- */
    double        dt(void) const { return dt_; }
    int           num(void) const { return num_; }
    const vect3  &p(void) const { return p_; }   /*< Δp_ij */
    const vect3  &v(void) const { return v_; }   /*< Δv_ij */
    const quat   &q(void) const { return q_; }   /*< Δq_ij */
    const vect3  &bg(void) const { return bg_; }
    const vect3  &ba(void) const { return ba_; }
    const vect3 &s(void)    const { return s_; }
    double       sodo(void) const { return sodo_; }
    const CovMatrix &cov(void) const { return cov_; }
    const JacMatrix &jac(void) const { return jac_; }

    /* 残余残差项的快捷访问（供图优化残差使用） */
    Eigen::Matrix3d R0(void) const { return qI.toEigen().toRotationMatrix(); }

private:
    void propagate(const ImuMeas &meas);
    /* 构造 F、G、Q */
    CovMatrix   buildF(const Eigen::Vector3d &a_body,
                       const Eigen::Vector3d &w_body,
                       const Eigen::Matrix3d &R,
                       double dt) const;
    GainMatrix  buildG(const Eigen::Matrix3d &R) const;
    NoiseMatrix buildQ(double dt) const;

    vect3  s_{O31};      /*< 预积分里程位移（b0 系） */
    double sodo_{0.0};   /*< 里程比例因子误差 */

    PreintegrationParam param_;
    std::deque<ImuMeas> meas_list_;  /*< 用于 repropagation 的历史测量 */

    /* ---- 累加状态 ---- */
    double dt_   = 0.0;
    int    num_  = 0;
    vect3  p_{O31};
    vect3  v_{O31};
    quat   q_{qI};

    /* ---- 线性化点处的零偏 ---- */
    vect3  bg_{O31};
    vect3  ba_{O31};

    /* ---- 中值积分缓存 ---- */
    vect3  last_dtheta_{O31};
    vect3  last_dvel_{O31};
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
    void addResidual(const mat &J, const vect &r);

    /* 边缘化前 num_marginalized 维。成功返回 true。
     * 若之前已 call 过 setPrior，会在原有先验基础上继续累积。 */
    bool marginalize(int num_marginalized);

    /* 把外部的先验（例如上一次边缘化的输出）作为新的残差加入。
     *   通常用于跨窗口传递：把 (J_out, r_out) 作为第一个残差块。 */
    void setPrior(const mat &J_out, const vect &r_out);

    /* 访问器 */
    int  dim(void) const { return n_; }
    const mat  &H(void) const { return H_; }
    const vect &b(void) const { return b_; }
    const mat  &J_prior(void) const { return J_prior_; }
    const vect &r_prior(void) const { return r_prior_; }

private:
    int  n_{0};
    mat  H_{};
    vect b_{};
    /* 上一次边缘化得到的先验 */
    mat  J_prior_{};
    vect r_prior_{};
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
    PreintResidual(const Preintegration &p, const vect3 &g, bool whiten, bool use_bias_jac)
        : preint_(p), g_n_(g) {
        /* 线性化点处的零偏 */
        bg_lin_ = p.bg().toEigen();
        ba_lin_ = p.ba().toEigen();

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
            const mat S = llt_sqrtinv(mat(p.cov()));
            if (S.row == 15) {
                S_ = S.toEigen();
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
    vect3         g_n_;
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
    GnssPosResidual(const vect3 &pos, const vect3 &std_dev)
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

    vect3 pos_, std_;
};


/* ============================================================================
 * 因子 3：GNSS 速度（3 维，增强版）
 *
 *   观测：vn（ENU，m/s）
 *   残差：r_i = (v_i - vn_i) / sigma_i，可加 Huber
 * ==========================================================================*/
struct GnssVelResidual {
    GnssVelResidual(const vect3 &vn, const vect3 &std_dev, double huber_delta = 0.0)
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

    vect3 vn_, std_;
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
 * 因子 6：边缘化先验（r 维）
 *   残差 = J_prior * δx - r_prior
 *   其中 δx 由"当前帧参数 - 线性化点参数"构成，这里简化为
 *   直接以切空间向量作为参数块（调用方需保证一致）。
 *   实用实现通常为每个被先验覆盖的参数块单独定义 residual，
 *   这里用动态维度版以展示接口。
 * ==========================================================================*/
class MarginalizationPriorFactor : public ceres::CostFunction {
public:
    MarginalizationPriorFactor(const mat &J, const vect &r)
        : J_(J), r_(r) {
        set_num_residuals(J.row);
        /* 列数 = 切空间维数（这里不含 sodo，纯 15 维）*/
        mutable_parameter_block_sizes()->push_back(J.clm);
    }

    bool Evaluate(const double *const *params,
                  double *residuals,
                  double **jacobians) const override {
        const double *dx = params[0];
        Eigen::Map<const Eigen::VectorXd> dx_e(dx, J_.clm);
        Eigen::Map<const Eigen::MatrixXd> J_e(J_.dd, J_.row, J_.clm);
        Eigen::Map<const Eigen::VectorXd> r_e(r_.dd, r_.rc);
        Eigen::Map<Eigen::VectorXd>       res(residuals, J_.row);

        res = J_e * dx_e - r_e;

        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
                Jout(jacobians[0], J_.row, J_.clm);
            Jout = J_e;
        }
        return true;
    }

private:
    mat  J_;
    vect r_;
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
    OdoVelResidual(double dS, double dt, const vect3 &omega_meas,
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
            T(omega_meas_.i) - bg(0),
            T(omega_meas_.j) - bg(1),
            T(omega_meas_.k) - bg(2));
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
    vect3  omega_meas_;
    double sigma_, huber_delta_;
};

/* ============================================================================
 * 因子 8：视觉重投影（2 维，手写雅可比，纯 mat3 / vect3 运算）
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
    VisualReprojResidual(const vect3& obs_i, const vect3& obs_j, double sigma)
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

        const vect3 p_i(pose_i[0], pose_i[1], pose_i[2]);
        const quat  q_i(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        const vect3 p_j(pose_j[0], pose_j[1], pose_j[2]);
        const quat  q_j(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);

        const vect3 t_bc(calib[7], calib[8], calib[9]);
        const quat  q_bc(calib[13], calib[10], calib[11], calib[12]);

        /* ---------- 2) 旋转矩阵 ---------- */
        const mat3 R_i  = q2mat(q_i);
        const mat3 R_j  = q2mat(q_j);
        const mat3 R_bc = q2mat(q_bc);

        /* ---------- 3) T_wc_i / T_wc_j ---------- */
        const mat3  R_wc_i = R_i * R_bc;
        const vect3 p_wc_i = p_i + R_i * t_bc;    /* mat3 * vect3 */

        const mat3  R_wc_j = R_j * R_bc;
        const vect3 p_wc_j = p_j + R_j * t_bc;

        /* ---------- 4) 特征点在相机 i 系下的 3D 坐标 ---------- */
        const double inv_rho = 1.0 / rho;
        const vect3  P_c_i(obs_i_.i * inv_rho,
                           obs_i_.j * inv_rho,
                           inv_rho);

        /* ---------- 5) 转世界系，再转相机 j 系 ---------- */
        const vect3 P_w   = p_wc_i + R_wc_i * P_c_i;
        const vect3 P_c_j = (~R_wc_j) * (P_w - p_wc_j);

        /* ---------- 6) 残差 ---------- */
        double Z = P_c_j.k;
        if (std::fabs(Z) < 1e-6) Z = 1e-6;
        const double Zinv = 1.0 / Z;
        const double X_Z  = P_c_j.i * Zinv;
        const double Y_Z  = P_c_j.j * Zinv;

        residuals[0] = (X_Z - obs_j_.i) * sigma_inv_;
        residuals[1] = (Y_Z - obs_j_.j) * sigma_inv_;

        if (jacobians == nullptr) return true;

        /* ---------- 7) 投影函数的雅可比：用两行 vect3 表示 ---------- */
        const vect3 J_pi_0(Zinv, 0.0, -X_Z * Zinv);
        const vect3 J_pi_1(0.0, Zinv, -Y_Z * Zinv);

        const mat3 R_wc_j_T = ~R_wc_j;

        /* ---------- 8) 对 pose_i 的雅可比（2×7 ambient） ---------- */
        if (jacobians[0] != nullptr) {
            /* 特征在 body_i 系下的坐标 */
            const vect3 P_i_Bi = t_bc + R_bc * P_c_i;

            /* ∂r/∂p_i = σ⁻¹ · J_pi · R_wc_j^T           (2×3) */
            const vect3 dpi_p0 = J_pi_0 * R_wc_j_T;
            const vect3 dpi_p1 = J_pi_1 * R_wc_j_T;

            /* ∂r/∂φ_i = -σ⁻¹ · J_pi · R_wc_j^T · R_i · [P_i_Bi]_×  (2×3) */
            const mat3 tmp     = R_wc_j_T * R_i * askew(P_i_Bi);
            const vect3 dpi_phi0 = J_pi_0 * tmp;
            const vect3 dpi_phi1 = J_pi_1 * tmp;

            /* 位置 3 列 */
            jacobians[0][0 * 7 + 0] = dpi_p0.i * sigma_inv_;
            jacobians[0][0 * 7 + 1] = dpi_p0.j * sigma_inv_;
            jacobians[0][0 * 7 + 2] = dpi_p0.k * sigma_inv_;
            jacobians[0][1 * 7 + 0] = dpi_p1.i * sigma_inv_;
            jacobians[0][1 * 7 + 1] = dpi_p1.j * sigma_inv_;
            jacobians[0][1 * 7 + 2] = dpi_p1.k * sigma_inv_;

            /* ∂r/∂q = 2·(∂r/∂δφ)·Q(q)^T，Q(q)^T 的 4 列按 [qx, qy, qz, qw] 排列：
            *   col(qx) = ( qw, -qz,  qy)
            *   col(qy) = ( qz,  qw, -qx)
            *   col(qz) = (-qy,  qx,  qw)
            *   col(qw) = (-qx, -qy, -qz) */
            const double qx = q_i.q1, qy = q_i.q2, qz = q_i.q3, qw = q_i.q0;
            const vect3 dphi_r0(-dpi_phi0.i, -dpi_phi0.j, -dpi_phi0.k);
            const vect3 dphi_r1(-dpi_phi1.i, -dpi_phi1.j, -dpi_phi1.k);
            const vect3 col_qx( qw, -qz,  qy);
            const vect3 col_qy( qz,  qw, -qx);
            const vect3 col_qz(-qy,  qx,  qw);
            const vect3 col_qw(-qx, -qy, -qz);

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
            const vect3 P_Bj = (~R_j) * (P_w - p_j);

            /* ∂r/∂p_j = -σ⁻¹ · J_pi · R_wc_j^T */
            const vect3 dpj_p0 = J_pi_0 * R_wc_j_T;
            const vect3 dpj_p1 = J_pi_1 * R_wc_j_T;

            /* ∂r/∂φ_j = σ⁻¹ · J_pi · R_bc^T · [P_Bj]_× */
            const mat3 tmp     = (~R_bc) * askew(P_Bj);
            const vect3 dpj_phi0 = J_pi_0 * tmp;
            const vect3 dpj_phi1 = J_pi_1 * tmp;

            jacobians[1][0*7+0] = -dpj_p0.i * sigma_inv_;
            jacobians[1][0*7+1] = -dpj_p0.j * sigma_inv_;
            jacobians[1][0*7+2] = -dpj_p0.k * sigma_inv_;
            jacobians[1][1*7+0] = -dpj_p1.i * sigma_inv_;
            jacobians[1][1*7+1] = -dpj_p1.j * sigma_inv_;
            jacobians[1][1*7+2] = -dpj_p1.k * sigma_inv_;

            /* ∂r/∂q_j = 2·(∂r/∂δφ_j)·Q(q_j)^T
            * Q(q_j)^T 的 4 列按 [qx, qy, qz, qw]：
            *   col(qx) = ( qw, -qz,  qy)
            *   col(qy) = ( qz,  qw, -qx)
            *   col(qz) = (-qy,  qx,  qw)
            *   col(qw) = (-qx, -qy, -qz)  */
            const double qx = q_j.q1, qy = q_j.q2, qz = q_j.q3, qw = q_j.q0;
            const vect3 col_qx( qw, -qz,  qy);
            const vect3 col_qy( qz,  qw, -qx);
            const vect3 col_qz(-qy,  qx,  qw);
            const vect3 col_qw(-qx, -qy, -qz);
            const vect3 dr0(dpj_phi0.i, dpj_phi0.j, dpj_phi0.k);
            const vect3 dr1(dpj_phi1.i, dpj_phi1.j, dpj_phi1.k);

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
            const mat3 Rd = R_wc_j_T * R_i - (~R_bc);

            const vect3 dt_0 = J_pi_0 * Rd;
            const vect3 dt_1 = J_pi_1 * Rd;

            jacobians[2][0*NUM_CALIB + 7] = dt_0.i * sigma_inv_;
            jacobians[2][0*NUM_CALIB + 8] = dt_0.j * sigma_inv_;
            jacobians[2][0*NUM_CALIB + 9] = dt_0.k * sigma_inv_;
            jacobians[2][1*NUM_CALIB + 7] = dt_1.i * sigma_inv_;
            jacobians[2][1*NUM_CALIB + 8] = dt_1.j * sigma_inv_;
            jacobians[2][1*NUM_CALIB + 9] = dt_1.k * sigma_inv_;

            /* --- 对 q_bc（calib[10..13]）---
             * ∂r/∂φ_bc = σ⁻¹ · J_pi · ([P_c_j]_× - R_wc_j^T · R_wc_i · [P_c_i]_×)
             */
            const mat3 inner = R_wc_j_T * R_wc_i * askew(P_c_i);
            const mat3 outer = askew(P_c_j) - inner;

            const vect3 dphi_0 = J_pi_0 * outer;
            const vect3 dphi_1 = J_pi_1 * outer;

            const double qbx = q_bc.q1, qby = q_bc.q2,
                        qbz = q_bc.q3, qbw = q_bc.q0;
            const vect3 col_qx( qbw, -qbz,  qby);
            const vect3 col_qy( qbz,  qbw, -qbx);
            const vect3 col_qz(-qby,  qbx,  qbw);
            const vect3 col_qw(-qbx, -qby, -qbz);
            const vect3 dr0(dphi_0.i, dphi_0.j, dphi_0.k);
            const vect3 dr1(dphi_1.i, dphi_1.j, dphi_1.k);

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
            const vect3 u = R_wc_j_T * (R_wc_i * P_c_i);   /* 3D 向量 */
            const double dr0 = dot(J_pi_0, u);
            const double dr1 = dot(J_pi_1, u);
            jacobians[3][0] = -dr0 * sigma_inv_ / rho;
            jacobians[3][1] = -dr1 * sigma_inv_ / rho;
        }

        return true;
    }

    vect3  obs_i_, obs_j_;
    double sigma_inv_;
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
    int    ant_mode           = ANT_MODE_LR_R;  /*< AntMode，见上 */
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
    vect3  odo_abv{0.0, 0.0, 0.0};    /*< 安装角 [pitch, 0, yaw]，rad */
    vect3  odo_lvOD{0.0, 0.0, 0.0};   /*< 杆臂（b 系，m） */
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
};

struct RunnerStats {
    int    n_frames       = 0;
    int    n_kf           = 0;      /*< 窗口残留关键帧 */
    int    n_kf_total     = 0;      /*< 累计处理的关键帧 */
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
        vect3  odo_abv{0.0, 0.0, 0.0};    /*< 安装角 [pitch, 0, yaw]，rad */
        vect3  odo_lvOD{0.0, 0.0, 0.0};   /*< 杆臂（b 系，m） */

        bool   use_visual    = false;
        double visual_sigma  = 1.5 / 460.0;
        double visual_huber  = 1.0;
        double inv_depth_min = 1.0 / 500.0;
        double inv_depth_max = 1.0 / 0.5;
    };

    GraphOptimizer() = default;
    explicit GraphOptimizer(const Options &opt) : opt_(opt) {}

    const Options &options(void) const { return opt_; }
    Options       &options(void) { return opt_; }

    /* 主入口 ---------------------------------------------------------------
     * 传入滑窗内所有关键帧，输出优化结果（原地更新 frames 的参数块）。
     * 若滑窗超限，会自动边缘化最旧帧并把先验保留在内部。 */
    bool optimize(std::vector<Frame> &frames, const vect3 &g_n);

    /* 访问器 */
    const mat  &J_prior(void) const { return J_prior_; }
    const vect &r_prior(void) const { return r_prior_; }
    bool  hasPrior(void) const { return J_prior_.row > 0; }
    void  clearPrior(void) { J_prior_ = mat(); r_prior_ = vect(); }

    void setCalibState(const CalibState &c) { c.toData(calib_data_.data()); }
    CalibState getCalibState() const { CalibState c; c.fromData(calib_data_.data()); return c; }

    std::map<int, double>&       invDepths()       { return inv_depths_; }
    const std::map<int, double>& invDepths() const { return inv_depths_; }

private:
    /* 组装 Ceres 问题：把 frames 的所有因子加进去 */
    void buildProblem(ceres::Problem &problem, std::vector<Frame> &frames,  const vect3 &g_n);

    /* 边缘化最旧帧：从 problem 中提取相关残差，Schur 补消去，
     * 输出新的先验 (J_prior_, r_prior_)。 */
    bool marginalizeOldestFrame(ceres::Problem &problem, std::vector<Frame> &frames, const mat  &J_prior_old, const vect  &r_prior_old);

    Options opt_;
    mat  J_prior_;   /*< 上一次边缘化的先验信息矩阵 */
    vect r_prior_;   /*< 上一次边缘化的先验信息向量 */
    std::array<double, NUM_CALIB> calib_data_{};
    std::map<int, double> inv_depths_;

public:
    /* 求解统计（供上层评估计算量与收敛性） */
    int    stat_solves{0};       /*< 调用 solve 的次数 */
    int    stat_converged{0};    /*< 正常收敛的次数 */
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
int IsZeros(const vect3 &v, double eps = EPS);
int IsZero(const double &val, double eps = EPS);
uint8_t IsZerosXY(const vect3 &v, double eps = EPS);
uint8_t IsNaN(const vect3 &v);
double diffYaw(double yaw, double yaw0);
vect3 abs(const vect3 &v);
vect3 maxabs(const vect3 &v1, const vect3 &v2);
double norm(const vect3 &v);
double normInf(const vect3 &v);
double normXY(const vect3 &v);
double normXYInf(const vect3 &v);
vect3 sqrt(const vect3 &v);
vect3 pow(const vect3 &v, int k);
double dot(const vect3 &v1, const vect3 &v2);
vect3 dotmul(const vect3 &v1, const vect3 &v2);
mat3 vxv(const vect3 &v1, const vect3 &v2);
double sinAng(const vect3 &v1, const vect3 &v2);
vect3 sort(const vect3 &v);
vect3 randn(const vect3 &mu, const vect3 &sigma);
double MKQt(double sR, double tau);
vect3 MKQt(const vect3 &sR, const vect3 &tau);

/* ============================ 融合 ============================ */
void fusion(double *x1, double *p1, const double *x2, const double *p2, int n = 9,
            double *xf = nullptr, double *pf = nullptr);
void fusion(vect3 &x1, vect3 &p1, const vect3 x2, const vect3 p2);
void fusion(vect3 &x1, vect3 &p1, const vect3 x2, const vect3 p2, vect3 &xf, vect3 &pf);



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
void accumulate_normal_equations(const mat &J, const vect &r, mat &H, vect &b);

/* Schur 补边缘化，返回 false 表示维数非法或数值分解失败 */
bool schur_complement(const mat &H, const vect &b, int num_marginalized, mat &J_out, vect &r_out);

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

#endif

