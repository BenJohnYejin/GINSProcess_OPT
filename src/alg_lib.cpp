/**
 * @file alg_lib.cpp
 * @brief 算法库实现
 *
 * 分节顺序与 alg_lib.h 的目录一致（自上而下）：
 *   1. 基础类型           Vector3d / Matrix3d / Quaterniond（动态向量/矩阵已统一为 Eigen）
 *   2. 标量与矢量工具       range / norm / diffYaw / askew ...
 *   3. 姿态与旋转          a2qua / rv2q / q2mat / m2att ...
 *   4. 线性代数            inv / llt_L / llt_sqrtinv / pinv_sym
 *   5. 地球模型与坐标系      earth / LocalFrame / xyz2blh ...
 *   6. IMU 与预积分         IMU::Update / Preintegration（中值积分 + 协方差传播）
 *   7. 状态与流形          State / plus / minus / applyDelta
 *   8. 边缘化              accumulate_normal_equations / schur_complement /
 *                          MarginalizationInfo
 *   9. 图优化              PoseManifold / 各残差因子 / GraphOptimizer
 *  10. 数据与运行入口       readSensorFile / StaticDetector / runRealData
 */

#include "alg_lib.h"
/* ---------- 全局常量 ---------- */

/* ============================================================================
 * Vector3d
 * ==========================================================================*/
/* ============================================================================
 * Matrix3d
 * ==========================================================================*/

/* 从 double / float 数组按行主序构造 ----------------------------------------------*/
/* 由三个矢量构造：isrow=1 时按行，否则按列 -----------------------------------------*/
/* v 加到对角线 -------------------------------------------------------------------*/
/* ============================================================================
 * Quaterniond
 * ==========================================================================*/

/* 失准角：与 ipos3g 一致，dq = quat * ~(*this) -----------------------------------*/
/* 用四元数旋转矢量：Cnb * v --------------------------------------------------------*/
void normlize(Eigen::Quaterniond &q) {
    double nq = std::sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
    /* 零范数保护：调用方可能传入未初始化的四元数（例如 calib 的 q_bc
     * 默认全 0 而忘了 setCalibState），此时 0/0 会传播成 NaN 并毁掉
     * 所有依赖四元数的残差。退化成单位四元数。 */
    if (!(nq > 1.0e-12)) {
        q.w() = 1.0; q.x() = 0.0; q.y() = 0.0; q.z() = 0.0;
        return;
    }
    q.w() /= nq;
    q.x() /= nq;
    q.y() /= nq;
    q.z() /= nq;
}







/* ============================================================================
 * 标量工具
 * ==========================================================================*/
double range(double val, double minVal, double maxVal) {
    if (val < minVal) {
        return minVal;
    }
    if (val > maxVal) {
        return maxVal;
    }
    return val;
}

int sign(double val, double eps) {
    if (val < -eps) {
        return -1;
    }
    if (val > eps) {
        return 1;
    }
    return 0;
}

double atan2Ex(double y, double x) {
    if (sign(y) == 0 && sign(x) == 0) {
        return 0.0;
    }
    return std::atan2(y, x);
}

double norm(const double *pd, int n) {
    return Eigen::Map<const Eigen::VectorXd>(pd, n).norm();
}

double norm1(const double *pd, int n) {
    return Eigen::Map<const Eigen::VectorXd>(pd, n).lpNorm<1>();
}

double normInf(const double *pd, int n) {
    return Eigen::Map<const Eigen::VectorXd>(pd, n).lpNorm<Eigen::Infinity>();
}

/* ============================================================================
 * Vector3d 工具
 * ==========================================================================*/

int IsZeros(const Eigen::Vector3d &v, double eps) {
    return (v.x() < eps && v.x() > -eps && v.y() < eps && v.y() > -eps && v.z() < eps && v.z() > -eps);
}

int IsZero(const double &val, double eps) { return (val < eps && val > -eps); }

uint8_t IsZerosXY(const Eigen::Vector3d &v, double eps) {
    return (v.x() < eps && v.x() > -eps && v.y() < eps && v.y() > -eps);
}

uint8_t IsNaN(const Eigen::Vector3d &v) {
    return (std::isnan(v.x()) || std::isnan(v.y()) || std::isnan(v.z())) ? 1 : 0;
}

double diffYaw(double yaw, double yaw0) {
    double dyaw = yaw - yaw0;
    if (dyaw >= PI) {
        dyaw -= _2PI;
    } else if (dyaw <= -PI) {
        dyaw += _2PI;
    }
    return dyaw;
}

Eigen::Vector3d abs(const Eigen::Vector3d &v) { return Eigen::Vector3d(v.cwiseAbs()); }

Eigen::Vector3d maxabs(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) {
    return Eigen::Vector3d(v1.cwiseAbs().cwiseMax(v2.cwiseAbs()));
}



double normXY(const Eigen::Vector3d &v) { return std::sqrt(v.x() * v.x() + v.y() * v.y()); }

double normXYInf(const Eigen::Vector3d &v) {
    double i = v.x() > 0 ? v.x() : -v.x(), j = v.y() > 0 ? v.y() : -v.y();
    return i > j ? i : j;
}

Eigen::Vector3d sqrt(const Eigen::Vector3d &v) { return Eigen::Vector3d(v.cwiseSqrt()); }

Eigen::Vector3d pow(const Eigen::Vector3d &v, int k) {
    Eigen::Vector3d pp = v;
    for (int i = 1; i < k; i++) {
        pp.x() *= v.x(), pp.y() *= v.y(), pp.z() *= v.z();
    }
    return pp;
}

double dot(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) { return v1.dot(v2); }

Eigen::Vector3d dotmul(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) {
    return Eigen::Vector3d(v1.cwiseProduct(v2));
}

Eigen::Matrix3d vxv(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) {
    return Eigen::Matrix3d(v1 * v2.transpose());
}

double sinAng(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2) {
    if (IsZeros(v1) || IsZeros(v2)) {
        return 0.0;
    }
    return norm(v1.cross(v2)) / (norm(v1) * norm(v2));
}

Eigen::Vector3d sort(const Eigen::Vector3d &v) {
    Eigen::Vector3d s = v;
    if (s.x() < s.y()) swapt(s.x(), s.y());
    if (s.x() < s.z()) swapt(s.x(), s.z());
    if (s.y() < s.z()) swapt(s.y(), s.z());
    return s;
}

std::mt19937 &rngEngine(void) {
    static std::mt19937 engine(20240918u); /*< 固定种子，保证结果可复现 */
    return engine;
}

Eigen::Vector3d randn(const Eigen::Vector3d &mu, const Eigen::Vector3d &sigma) {
    std::normal_distribution<double> dist(0.0, 1.0);
    return Eigen::Vector3d(mu.x() + sigma.x() * dist(rngEngine()), mu.y() + sigma.y() * dist(rngEngine()),
                 mu.z() + sigma.z() * dist(rngEngine()));
}

double MKQt(double sR, double tau) { return sR * sR * 2.0 / tau; }

Eigen::Vector3d MKQt(const Eigen::Vector3d &sR, const Eigen::Vector3d &tau) {
    return Eigen::Vector3d(sR.x() * sR.x() * 2.0 / tau.x(), sR.y() * sR.y() * 2.0 / tau.y(), sR.z() * sR.z() * 2.0 / tau.z());
}

/* ============================================================================
 * 姿态 / 旋转
 * ==========================================================================*/

/* 旋转矢量 -> 四元数（小角度用泰勒展开，大角度用 Eigen::AngleAxis） ---------------*/
Eigen::Quaterniond rv2q(const Eigen::Vector3d &rv) {
    double n2 = rv.x() * rv.x() + rv.y() * rv.y() + rv.z() * rv.z();
    if (n2 < (DEG * DEG)) {
        double n4 = n2 * n2;
        double c  = 1.0 - n2 * (1.0 / 8.0) + n4 * (1.0 / 384.0);   /*< cos(|rv|/2) */
        double f  = 0.5 - n2 * (1.0 / 48.0) + n4 * (1.0 / 3840.0); /*< sin(|rv|/2)/|rv| */
        return Eigen::Quaterniond(c, f * rv.x(), f * rv.y(), f * rv.z());
    }
    double th = std::sqrt(n2);
    return Eigen::Quaterniond(Eigen::Quaterniond(Eigen::AngleAxisd(th, rv / th)));
}

Eigen::Matrix3d rv2m(const Eigen::Vector3d &rv) { return q2mat(rv2q(rv)); }

Eigen::Matrix3d q2mat(const Eigen::Quaterniond &qnb) { return Eigen::Matrix3d(qnb.toRotationMatrix()); }

Eigen::Quaterniond m2qua(const Eigen::Matrix3d &Cnb) {
    Eigen::Quaterniond q(Cnb);
    q.normalize();
    /* 与 ipos3g 一致：固定 q0 >= 0 的分支 */
    if (q.w() < 0) {
        q.coeffs() = -q.coeffs();
    }
    return Eigen::Quaterniond(q);
}

Eigen::Matrix3d askew(const Eigen::Vector3d &v) {
    /*
     *      [  0  -k   j ]
     *      [  k   0  -i ]
     *      [ -j   i   0 ]
     */
    return (Eigen::Matrix3d() << 0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0).finished();
}

/* 欧拉角 [pitch,roll,yaw] -> Cnb（ipos3g / PSINS 约定，保留闭式） -----------------*/
Eigen::Matrix3d a2mat(const Eigen::Vector3d &att) {
    double si = std::sin(att.x()), ci = std::cos(att.x()), sj = std::sin(att.y()), cj = std::cos(att.y()),
           sk = std::sin(att.z()), ck = std::cos(att.z());
    return (Eigen::Matrix3d() << cj * ck - si * sj * sk, -ci * sk, sj * ck + si * cj * sk, cj * sk + si * sj * ck, ci * ck, sj * sk - si * cj * ck, -ci * sj, si, ci * cj).finished();
}

Eigen::Vector3d m2att(const Eigen::Matrix3d &Cnb) {
    return Eigen::Vector3d(asinEx(Cnb(2,1)), atan2Ex(-Cnb(2,0), Cnb(2,2)), atan2Ex(-Cnb(0,1), Cnb(1,1)));
}

/* 反序欧拉角 -> Cnb（保留闭式） ---------------------------------------------------*/
Eigen::Matrix3d ar2mat(const Eigen::Vector3d &attr) {
    double si = std::sin(attr.x()), ci = std::cos(attr.x()), sj = std::sin(attr.y()), cj = std::cos(attr.y()),
           sk = std::sin(attr.z()), ck = std::cos(attr.z());
    return (Eigen::Matrix3d() << cj * ck, si * sj * ck - ci * sk, ci * sj * ck + si * sk, cj * sk, si * sj * sk + ci * ck, ci * sj * sk - si * ck, -sj, si * cj, ci * cj).finished();
}

Eigen::Quaterniond ar2qua(const Eigen::Vector3d &attr) { return m2qua(ar2mat(attr)); }

Eigen::Vector3d m2attr(const Eigen::Matrix3d &Cnb) {
    return Eigen::Vector3d(atan2Ex(Cnb(2,1), Cnb(2,2)), asinEx(-Cnb(2,0)), atan2Ex(Cnb(1,0), Cnb(0,0)));
}

Eigen::Vector3d q2attr(const Eigen::Quaterniond &qnb) { return m2attr(q2mat(qnb)); }

/* 欧拉角 [pitch,roll,yaw] -> 四元数 -----------------*/
Eigen::Quaterniond a2qua(double pitch, double roll, double yaw) {
    pitch /= 2.0, roll /= 2.0, yaw /= 2.0;
    double sp = std::sin(pitch), sr = std::sin(roll), sy = std::sin(yaw),
           cp = std::cos(pitch), cr = std::cos(roll), cy = std::cos(yaw);
    return Eigen::Quaterniond(cp * cr * cy - sp * sr * sy,
                sp * cr * cy - cp * sr * sy,
                cp * sr * cy + sp * cr * sy,
                cp * cr * sy + sp * sr * cy);
}

Eigen::Quaterniond a2qua(const Eigen::Vector3d &att) { return a2qua(att.x(), att.y(), att.z()); }

Eigen::Vector3d q2att(const Eigen::Quaterniond &qnb) {
    double q11 = qnb.w() * qnb.w(), q12 = qnb.w() * qnb.x(), q13 = qnb.w() * qnb.y(),
           q14 = qnb.w() * qnb.z(), q22 = qnb.x() * qnb.x(), q23 = qnb.x() * qnb.y(),
           q24 = qnb.x() * qnb.z(), q33 = qnb.y() * qnb.y(), q34 = qnb.y() * qnb.z(),
           q44 = qnb.z() * qnb.z();
    return Eigen::Vector3d(asinEx(2 * (q34 + q12)), atan2Ex(-2 * (q24 - q13), q11 - q22 - q33 + q44),
                 atan2Ex(-2 * (q23 - q14), q11 - q22 + q33 - q44));
}

/* 设置偏航角（保留 pitch/roll）——原 quat::SetYaw 的替代 */
void setYaw(Eigen::Quaterniond &q, double yaw) {
    Eigen::Vector3d att = q2att(q);
    att.z() = yaw;
    q = a2qua(att);
}
#if 0
/* 四元数 -> 旋转矢量 --------------------------------------------------------------*/
Eigen::Vector3d q2rv(const Eigen::Quaterniond &q) {
    Eigen::Quaterniond dq = q;
    if (dq.w() < 0) {
        dq.w() = -dq.w(), dq.x() = -dq.x(), dq.y() = -dq.y(), dq.z() = -dq.z();
    }
    if (dq.w() > 1.0) {
        dq.w() = 1.0;
    }
    double n2 = std::acos(dq.w()), f;
    if (n2 > 1.0e-20) {
        f = 2.0 / (std::sin(n2) / n2);
    } else {
        f = 2.0;
    }
    return Eigen::Vector3d(dq.x(), dq.y(), dq.z()) * f;
}
#endif
Eigen::Vector3d q2rv(const Eigen::Quaterniond &q) {
    Eigen::Quaterniond dq = q;
    double n = std::sqrt(dq.w()*dq.w() + dq.x()*dq.x() + dq.y()*dq.y() + dq.z()*dq.z());
    if (n > 0.0) { dq.w() /= n; dq.x() /= n; dq.y() /= n; dq.z() /= n; }
    /* 规范化：让 q0 >= 0；若 q0 == 0，让首个非零虚部 >= 0 */
    if (dq.w() < 0.0 ||  (dq.w() == 0.0 &&(dq.x() < 0.0 || (dq.x() == 0.0 && (dq.y() < 0.0 || (dq.y() == 0.0 && dq.z() < 0.0)))))) {
        dq.w() = -dq.w(); dq.x() = -dq.x(); dq.y() = -dq.y(); dq.z() = -dq.z();
    }
    double s = std::sqrt(std::max(0.0, 1.0 - dq.w() * dq.w()));
    double theta = 2.0 * std::acos(range(dq.w(), -1.0, 1.0));
    double k = (theta < 1.0e-20) ? 2.0 : theta / s;
    return Eigen::Vector3d(dq.x() * k, dq.y() * k, dq.z() * k);
}

/* 旋转矩阵 -> 旋转矢量 ------------------------------------------------------------*/
Eigen::Vector3d m2rv(const Eigen::Matrix3d &Cnb) {
    double phi = std::acos(range((trace(Cnb) - 1.0) / 2.0, -1.0, 1.0));
    double afa = (-1.0e-10 < phi && phi < 1.0e-10) ? 0.5 : phi / (2.0 * std::sin(phi));
    return Eigen::Vector3d(Cnb(2,1) - Cnb(1,2), Cnb(0,2) - Cnb(2,0), Cnb(1,0) - Cnb(0,1)) * afa;
}

Eigen::Vector3d qq2phi(const Eigen::Quaterniond &qcalcu, const Eigen::Quaterniond &qreal) { return q2rv(qreal * ((qcalcu.conjugate()))); }

Eigen::Quaterniond addmu(const Eigen::Quaterniond &q, const Eigen::Vector3d &mu) { return q * rv2q(mu); }

Eigen::Quaterniond UpDown(const Eigen::Quaterniond &q) {
    Eigen::Vector3d att = q2att(q);
    att.x()     = -att.x();
    att.y() += PI;
    return a2qua(att);
}

Eigen::Vector3d sv2att(const Eigen::Vector3d &fb, double yaw0, const Eigen::Vector3d &fn) {
    Eigen::Vector3d phi = fb.cross(fn);
    double afa = std::acos(range(dot(fn, fb) / norm(fn) / norm(fb), -1.0, 1.0)), nphi = norm(phi);
    Eigen::Vector3d att = q2att(rv2q(phi * (nphi < 1.0e-10 ? 1.0 : afa / nphi)));
    att.z()     = yaw0;
    return att;
}

Eigen::Vector3d vn2att(const Eigen::Vector3d &vn) {
    double vel = normXY(vn);
    if (vel < 1.0e-6) {
        return O31;
    }
    return Eigen::Vector3d(std::atan2(vn.z(), vel), 0, std::atan2(-vn.x(), vn.y()));
}

double vn2att(double vel_east, double vel_north) {
    double vel = vel_east * vel_east + vel_north * vel_north;
    if (vel < 3) {
        return 0;
    }
    return std::atan2(-vel_east, vel_north);
}

/* ============================================================================
 * Matrix3d 线性代数
 * ==========================================================================*/

Eigen::Matrix3d Rot(double angle, char axis) {
    double s = std::sin(angle), c = std::cos(angle);
    switch (axis) {
    case 'x':
    case 'X': return (Eigen::Matrix3d() << 1, 0, 0, 0, c, -s, 0, s, c).finished();
    case 'y':
    case 'Y': return (Eigen::Matrix3d() << c, 0, s, 0, 1, 0, -s, 0, c).finished();
    default: return (Eigen::Matrix3d() << c, -s, 0, s, c, 0, 0, 0, 1).finished();
    }
}

Eigen::Matrix3d rcijk(const Eigen::Matrix3d &m, int ijk) {
    switch (ijk) {
    case 021: return (Eigen::Matrix3d() << m(0,0), m(0,2), m(0,1), m(2,0), m(2,2), m(2,1), m(1,0), m(1,2), m(1,1)).finished();
    case 102: return (Eigen::Matrix3d() << m(1,1), m(1,0), m(1,2), m(0,1), m(0,0), m(0,2), m(2,1), m(2,0), m(2,2)).finished();
    case 120: return (Eigen::Matrix3d() << m(1,1), m(1,2), m(1,0), m(2,1), m(2,2), m(2,0), m(0,1), m(0,2), m(0,0)).finished();
    case 201: return (Eigen::Matrix3d() << m(2,2), m(2,0), m(2,1), m(0,2), m(0,0), m(0,1), m(1,2), m(1,0), m(1,1)).finished();
    case 210: return (Eigen::Matrix3d() << m(2,2), m(2,1), m(2,0), m(1,2), m(1,1), m(1,0), m(0,2), m(0,1), m(0,0)).finished();
    default: return m;
    }
}

double trMMT(const Eigen::Matrix3d &m1, const Eigen::Matrix3d &m2) {
    const Eigen::Matrix3d &b = (&m2 == &I33) ? m1 : m2;
    return (m1.cwiseProduct(b)).sum();
}

void symmetry(Eigen::Matrix3d &m) {
    m(0,1) = m(1,0) = (m(0,1) + m(1,0)) * 0.5;
    m(0,2) = m(2,0) = (m(0,2) + m(2,0)) * 0.5;
    m(1,2) = m(2,1) = (m(1,2) + m(2,1)) * 0.5;
}

Eigen::Matrix3d pow(const Eigen::Matrix3d &m, int k) {
    Eigen::Matrix3d mm = m;
    for (int i = 1; i < k; i++) {
        mm = mm * m;
    }
    return mm;
}

double trace(const Eigen::Matrix3d &m) { return m.trace(); }

double det(const Eigen::Matrix3d &m) { return m.determinant(); }

Eigen::Matrix3d adj(const Eigen::Matrix3d &m) {
    /* 不用 Eigen 的 adjugate()：
     *   1) 它靠 EIGEN_MATRIXBASE_PLUGIN 注入到 MatrixBase，
     *      对 Map<> 类型的可见性在不同 3.4.x 版本/包含顺序下不一致；
     *   2) 3x3 伴随矩阵直接展开也就 9 个乘减，没必要依赖库实现。
     * 伴随矩阵定义：A·adj(A) = det(A)·I，元素为代数余子式的转置。 */
    const Eigen::Matrix3d A = m;     /* Map -> 具体矩阵（列主序也可） */
    Eigen::Matrix3d a;
    a(0,0) = A(1,1)*A(2,2) - A(1,2)*A(2,1);
    a(0,1) = A(0,2)*A(2,1) - A(0,1)*A(2,2);
    a(0,2) = A(0,1)*A(1,2) - A(0,2)*A(1,1);
    a(1,0) = A(1,2)*A(2,0) - A(1,0)*A(2,2);
    a(1,1) = A(0,0)*A(2,2) - A(0,2)*A(2,0);
    a(1,2) = A(0,2)*A(1,0) - A(0,0)*A(1,2);
    a(2,0) = A(1,0)*A(2,1) - A(1,1)*A(2,0);
    a(2,1) = A(0,1)*A(2,0) - A(0,0)*A(2,1);
    a(2,2) = A(0,0)*A(1,1) - A(0,1)*A(1,0);
    return Eigen::Matrix3d(a);
}

Eigen::Matrix3d inv(const Eigen::Matrix3d &m) { return Eigen::Matrix3d(m.inverse()); }

Eigen::Vector3d diag(const Eigen::Matrix3d &m) { return Eigen::Vector3d(m(0,0), m(1,1), m(2,2)); }

Eigen::Matrix3d diag(const Eigen::Vector3d &v) { return (Eigen::Matrix3d() << v.x(), 0, 0, 0, v.y(), 0, 0, 0, v.z()).finished(); }

Eigen::Matrix3d askew(const Eigen::Matrix3d &m, int I) {
    Eigen::Matrix3d m1;
    m1(0,1) = (m(0,1) - m(1,0)) / 2;
    m1(0,2) = (m(0,2) - m(2,0)) / 2;
    m1(1,2) = (m(1,2) - m(2,1)) / 2;
    m1(1,0) = -m1(0,1);
    m1(2,0) = -m1(0,2);
    m1(2,1) = -m1(1,2);
    if (I == 0) {
        m1(0,0) = m1(1,1) = m1(2,2) = 0.0;
    } else if (I == 1) {
        m1(0,0) = m1(1,1) = m1(2,2) = 1.0;
    } else {
        m1(0,0) = m(0,0), m1(1,1) = m(1,1), m1(2,2) = m(2,2);
    }
    return m1;
}

Eigen::Matrix3d dotmul(const Eigen::Matrix3d &m1, const Eigen::Matrix3d &m2) { return Eigen::Matrix3d(m1.cwiseProduct(m2)); }

Eigen::Matrix3d MMT(const Eigen::Matrix3d &m1, const Eigen::Matrix3d &m2) {
    const Eigen::Matrix3d &b = (&m2 == &I33) ? m1 : m2;
    return Eigen::Matrix3d(m1 * b.transpose());
}


Eigen::Matrix3d randn(const Eigen::Matrix3d &mu, const double &sigma) {
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::Matrix3d r = mu;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            r(i, j) += sigma * dist(rngEngine());
        }
    }
    return Eigen::Matrix3d(r);
}

/* ============================================================================
 * 动态向量线性代数（Eigen::VectorXd）
 * ==========================================================================*/

Eigen::VectorXd absv(const Eigen::VectorXd &v) { return v.cwiseAbs(); }




Eigen::VectorXd pow(const Eigen::VectorXd &v, int k) {
    Eigen::VectorXd pp = v;
    for (int i = 1; i < k; i++) {
        pp = pp.cwiseProduct(v);
    }
    return pp;
}

/* 注意：与旧实现一致，按**降序**排序 */
Eigen::VectorXd sort(const Eigen::VectorXd &v) {
    Eigen::VectorXd vtmp = v;
    std::sort(vtmp.data(), vtmp.data() + vtmp.size(), std::greater<double>());
    return vtmp;
}

double dot(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2) {
    assert(v1.size() == v2.size());
    return v1.dot(v2);
}

Eigen::VectorXd dotmul(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2) {
    assert(v1.size() == v2.size());
    return v1.cwiseProduct(v2);
}

Eigen::VectorXd randn(const Eigen::VectorXd &mu, const Eigen::VectorXd &sigma) {
    assert(mu.size() == sigma.size());
    Eigen::VectorXd res(mu.size());
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < res.size(); i++) {
        res(i) = mu(i) + sigma(i) * dist(rngEngine());
    }
    return res;
}

/* ============================================================================
 * 动态矩阵线性代数（Eigen::MatrixXd）
 * ==========================================================================*/

void symmetry(Eigen::MatrixXd &m) {
    assert(m.rows() == m.cols());
    m = 0.5 * (m + m.transpose());
}

double trace(const Eigen::MatrixXd &m) {
    assert(m.rows() == m.cols());
    return m.trace();
}



Eigen::MatrixXd dotmul(const Eigen::MatrixXd &m1, const Eigen::MatrixXd &m2) {
    assert(m1.rows() == m2.rows() && m1.cols() == m2.cols());
    return m1.cwiseProduct(m2);
}

Eigen::VectorXd diag(const Eigen::MatrixXd &m) {
    /* 与旧实现一致：对"高瘦"矩阵（rows > cols）超出部分补 0 */
    const int n = static_cast<int>(std::min(m.rows(), m.cols()));
    Eigen::VectorXd v = Eigen::VectorXd::Zero(m.rows());
    for (int i = 0; i < n; i++) v(i) = m(i, i);
    return v;
}

Eigen::MatrixXd diag(const Eigen::VectorXd &v) {
    return v.asDiagonal();
}

Eigen::MatrixXd eye(int n) { return Eigen::MatrixXd::Identity(n, n); }

Eigen::MatrixXd inv4(const Eigen::MatrixXd &m) {
    assert(m.rows() == 4 && m.cols() == 4);
    return m.inverse();
}




Eigen::MatrixXd randn(const Eigen::MatrixXd &mu, const double &sigma) {
    Eigen::MatrixXd res(mu.rows(), mu.cols());
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < mu.size(); i++) {
        res(i) = mu(i) + sigma * dist(rngEngine());
    }
    return res;
}

/* ============================================================================
 * 坐标系
 * ==========================================================================*/

Eigen::Matrix3d pos2Cen(const Eigen::Vector3d &pos) {
    double si = std::sin(pos.x()), ci = std::cos(pos.x()), sj = std::sin(pos.y()), cj = std::cos(pos.y());
    return (Eigen::Matrix3d() << -sj, -si * cj, ci * cj, cj, -si * sj, ci * sj, 0, ci, si).finished();
}

Eigen::Vector3d xyz2blh(const Eigen::Vector3d &xyz) {
    double s     = normXY(xyz);
    /*
     * Bowring/Heiskanen 参数纬度：theta = atan2(z*a, b*p)，p = sqrt(x^2+y^2) = s
     */
    double theta = std::atan2(xyz.z() * RE, s * RP);
    double s3    = std::sin(theta), c3 = std::cos(theta);
    s3 = s3 * s3 * s3;
    c3 = c3 * c3 * c3;
    if (s < 1.0) { return O31; }
    double L  = std::atan2(xyz.y(), xyz.x());
    double B  = std::atan2(xyz.z() + (ep2 * RP * s3), s - e2 * RE * c3);
    double sB = std::sin(B), cB = std::cos(B);
    double N  = RE / std::sqrt(1 - e2 * sB * sB);
    return Eigen::Vector3d(B, L, s / cB - N);
}

Eigen::Vector3d blh2xyz(const Eigen::Vector3d &blh) {
    double sB = std::sin(blh.x()), cB = std::cos(blh.x()), sL = std::sin(blh.y()), cL = std::cos(blh.y()),
           N = RE / std::sqrt(1 - e2 * sB * sB);
    return Eigen::Vector3d((N + blh.z()) * cB * cL, (N + blh.z()) * cB * sL, (N * (1 - e2) + blh.z()) * sB);
}

/* 旧 vect3*mat3 是行向量右乘 VᵀM，等价写成 (Mᵀ·V) */
Eigen::Vector3d Vxyz2enu(const Eigen::Vector3d &Vxyz, const Eigen::Vector3d &pos) {
    return pos2Cen(pos).transpose() * Vxyz;
}

/* ============================================================================
 * 融合（Kalman 融合公式）
 * ==========================================================================*/

void fusion(double *x1, double *p1, const double *x2, const double *p2, int n, double *xf,
            double *pf) {
    if (xf == nullptr) {
        xf = x1;
        pf = p1;
    }
    double *x10 = nullptr, *xf0 = nullptr;
    Eigen::Vector3d att1;

    /* n < 100 时前 3 个量按"姿态"处理（与 ipos3g 一致，含对 x2 的写回） */
    if (n < 100) {
        x10  = x1;
        xf0  = xf;
        att1 = Eigen::Vector3d(x1[0], x1[1], x1[2]);
        Eigen::Vector3d phi = qq2phi(a2qua(Eigen::Vector3d(x2[0], x2[1], x2[2])), a2qua(att1));
        double *x2w = const_cast<double *>(x2);
        x2w[0] = phi.x();
        x2w[1] = phi.y();
        x2w[2] = phi.z();
        x1[0] = x1[1] = x1[2] = 0.0;
    }

    int j = (n > 100) ? 100 : 0;
    for (; j < n; j++, x1++, p1++, x2++, p2++, xf++, pf++) {
        double p1p2 = *p1 + *p2;
        *xf = (*p1 * (*x2) + *p2 * (*x1)) / p1p2;
        *pf = *p1 * *p2 / p1p2;
    }

    if (n < 100 && xf0 != nullptr) {
        Eigen::Vector3d v = q2att(qPlusPhi(a2qua(att1), Eigen::Vector3d(xf0[0], xf0[1], xf0[2])));
        xf0[0] = v.x();
        xf0[1] = v.y();
        xf0[2] = v.z();
        if (xf0 != x10) {
            x10[0] = att1.x();
            x10[1] = att1.y();
            x10[2] = att1.z();
        }
    }
}

void fusion(Eigen::Vector3d &x1, Eigen::Vector3d &p1, const Eigen::Vector3d x2, const Eigen::Vector3d p2) {
    fusion(&x1.x(), &p1.x(), &x2.x(), &p2.x(), 3);
}

void fusion(Eigen::Vector3d &x1, Eigen::Vector3d &p1, const Eigen::Vector3d x2, const Eigen::Vector3d p2, Eigen::Vector3d &xf, Eigen::Vector3d &pf) {
    fusion(&x1.x(), &p1.x(), &x2.x(), &p2.x(), 3, &xf.x(), &pf.x());
}

/* ============================================================================
 * 扩展：对称正定线性代数（Eigen）
 * ==========================================================================*/

Eigen::MatrixXd llt_L(const Eigen::MatrixXd &A) {
    if (A.rows() != A.cols() || A.rows() == 0) {
        return Eigen::MatrixXd();
    }
    /* 注意必须 .eval()：Eigen 对 `a = a + a.transpose()` 这种自别名写法
     * 不会自动插入临时变量，逐个元素赋值会读到已被覆盖的转置项。 */
    Eigen::MatrixXd a = (0.5 * (A + A.transpose())).eval();

    /* 只在真正正定时返回下三角 Cholesky 因子 L（A = L·Lᵀ）。
     *
     * 【不要把这里退化成「LDLT + 把负的 D 钳到小正数」】
     * 那样拼出来的 Pᵀ·L·√D 虽然近似满足 L·Lᵀ = A，但**一般不是下三角**，
     * 而 llt_sqrtinv / llt_inv / schur_complement 都用
     * `triangularView<Eigen::Lower>().solve(...)` 消费本函数的返回值 ——
     * 对非三角矩阵做三角求解会静默给出错误结果（triangularView 的下三角
     * 部分可能奇异，还会产生 NaN）。
     *
     * 实测：600 个秩亏/含极小负特征值的对称矩阵里有 300 个走钳位分支并返回
     * 非下三角矩阵；受控实验 200 例中 schur_complement 的 r_out 全部为 NaN。
     *
     * 因此非正定时一律返回空矩阵，让调用方走各自的特征分解 / 伪逆兜底路径
     * （llt_sqrtinv 与 schur_complement 都已有正确的兜底分支）。 */
    Eigen::LLT<Eigen::MatrixXd> llt(a);
    if (llt.info() != Eigen::Success) {
        return Eigen::MatrixXd();
    }
    const Eigen::MatrixXd Lm = llt.matrixL();

    /* 数值奇异保护：
     * 对"舍入后恰好正定"的矩阵（例如由 Q·diag(0,λ2,λ3)·Qᵀ 构造出来的、
     * 数学上秩亏但浮点下最小特征值变成 ~1e-16·dmax 的协方差），LLT 也会
     * 报告成功，而它的最小主元只有 ~1e-8·最大主元。此时
     *   - llt_sqrtinv 会返回 L⁻¹，在该方向上的增益达 1e8；
     *   - 白化后残差被放大 1e16 量级，等于把一个不可观方向当成强约束。
     * 所以这里用 Cholesky 主元的相对大小做条件数保护
     * （cond(A) ≈ (dmax/dmin)²，阈值 1e-5 对应 cond ≲ 1e10），
     * 不满足就返回空，让调用方走"相对阈值截断的伪逆"路径。
     * 良态协方差（cond ~1e3）的 dmin/dmax ≈ 0.03，仍走快路径。 */
    const double dmn = Lm.diagonal().cwiseAbs().minCoeff();
    const double dmx = Lm.diagonal().cwiseAbs().maxCoeff();
    if (!(dmx > 0.0) || !(dmn > 1.0e-5 * dmx)) {
        return Eigen::MatrixXd();
    }
    return Lm;
}


Eigen::MatrixXd llt_sqrtinv(const Eigen::MatrixXd &A) {
    Eigen::MatrixXd L = llt_L(A);
    if (L.rows() > 0) {
        Eigen::MatrixXd Le   = L;
        Eigen::MatrixXd Linv = Le.triangularView<Eigen::Lower>().solve(
            Eigen::MatrixXd::Identity(Le.rows(), Le.cols()));
        /* S = L^{-1}（下三角），S^T S = L^{-T} L^{-1} = A^{-1} */
        if (Linv.allFinite()) {
            return Linv;   // 原为 Linv.transpose()
        }
    }

    /* 兜底：A 严重退化时（例如刚 reset、协方差里还有数值负特征值），
     * 用特征分解求带相对截断的伪逆平方根：S = V·diag(1/sqrt(d))·Vᵀ。
     * 与 pinv_sym / schur_complement 的处理方式保持一致。 */
    if (A.rows() != A.cols() || A.rows() == 0) {
        return Eigen::MatrixXd();
    }
    Eigen::MatrixXd a = (0.5 * (A + A.transpose())).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(a);
    if (es.info() != Eigen::Success) {
        return Eigen::MatrixXd();
    }
    const Eigen::VectorXd d = es.eigenvalues();
    const Eigen::MatrixXd V = es.eigenvectors();
    const double dmax       = d.cwiseAbs().maxCoeff();
    if (!(dmax > 0.0) || !std::isfinite(dmax)) {
        return Eigen::MatrixXd();
    }
    const double thr = 1.0e-10 * dmax;
    Eigen::VectorXd dsinv(d.size());
    for (int i = 0; i < d.size(); i++) {
        dsinv(i) = (d(i) > thr) ? 1.0 / std::sqrt(d(i)) : 0.0;
    }
    return V * dsinv.asDiagonal() * V.transpose();
}

Eigen::MatrixXd llt_inv(const Eigen::MatrixXd &A) {
    Eigen::MatrixXd L = llt_L(A);
    if (L.rows() == 0) {
        return Eigen::MatrixXd();
    }
    Eigen::MatrixXd Le = L;
    Eigen::MatrixXd Linv = Le.triangularView<Eigen::Lower>().solve(
        Eigen::MatrixXd::Identity(Le.rows(), Le.cols()));
    return Linv.transpose() * Linv;
}

Eigen::MatrixXd pinv_sym(const Eigen::MatrixXd &A, double eps) {
    if (A.rows() != A.cols() || A.rows() == 0) {
        return Eigen::MatrixXd();
    }
    Eigen::MatrixXd a = A;
    a                 = (0.5 * (a + a.transpose())).eval(); /*< 必须 eval，见 llt_L 的注释 */

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(a);
    if (es.info() != Eigen::Success) {
        return Eigen::MatrixXd();
    }
    Eigen::VectorXd d = es.eigenvalues();
    Eigen::MatrixXd V = es.eigenvectors();

    /* 相对阈值：以最大特征值为基准 */
    double dmax = d.cwiseAbs().maxCoeff();
    double thr  = eps * std::max(dmax, 1e-300);
    Eigen::VectorXd dinv(d.size());
    for (int i = 0; i < d.size(); i++) {
        dinv(i) = (std::fabs(d(i)) > thr) ? 1.0 / d(i) : 0.0;
    }
    return V * dinv.asDiagonal() * V.transpose();
}


void LocalFrame::init(const Eigen::Vector3d &blh0) {
    blh0_ = blh0;
    eth_.Update(blh0_, O31);
    gravity_ = eth_.gn;
    init_    = true;

    const double sl = std::sin(blh0.x()), cl = std::cos(blh0.y() != 0.0 ? blh0.y() : 0.0);
    /* 实际应写成： */
    const double lat = blh0.x(), lon = blh0.y();
    const double sl_ = std::sin(lat), cl_ = std::cos(lat);
    const double so_ = std::sin(lon), co_ = std::cos(lon);
    R_en_m3_ << -so_, -sl_ * co_,  cl_ * co_,
                 co_, -sl_ * so_,  cl_ * so_,
                 0.0,        cl_,        sl_;
    p0_ecef_v3_ = blh2xyz(blh0);

}

Eigen::Vector3d LocalFrame::blh2neu(const Eigen::Vector3d &blh) const {
    double pn = (blh.x() - blh0_.x()) * eth_.RMh;
    double pe = (blh.y() - blh0_.y()) * eth_.clRNh;
    double pu = blh.z() - blh0_.z();
    return Eigen::Vector3d(pe, pn, pu);
}

Eigen::Vector3d LocalFrame::neu2blh(const Eigen::Vector3d &neu) const {
    double lat = blh0_.x() + neu.y() * eth_.f_RMh;
    double lon = blh0_.y() + neu.x() * eth_.f_clRNh;
    double hgt = blh0_.z() + neu.z();
    return Eigen::Vector3d(lat, lon, hgt);
}

Eigen::Vector3d quatVec(const Eigen::Quaterniond &q) { return Eigen::Vector3d(q.x(), q.y(), q.z()); }

/* pose = {pE, pN, pU, qx, qy, qz, qw}
 * mix  = {v, bg, ba} 或 {v, bg, ba, sodo} ----------------------------------------*/
void stateToData(const State &state, double *pose, double *mix, bool with_odometer) {
    pose[0] = state.p.x();
    pose[1] = state.p.y();
    pose[2] = state.p.z();
    pose[3] = state.q.x(); /*< x */
    pose[4] = state.q.y(); /*< y */
    pose[5] = state.q.z(); /*< z */
    pose[6] = state.q.w(); /*< w */

    mix[0] = state.v.x();
    mix[1] = state.v.y();
    mix[2] = state.v.z();
    mix[3] = state.bg.x();
    mix[4] = state.bg.y();
    mix[5] = state.bg.z();
    mix[6] = state.ba.x();
    mix[7] = state.ba.y();
    mix[8] = state.ba.z();
    (void)with_odometer;   /* mix 现为 9 维；sodo 由 calib[0] 承载 */
}

void stateFromData(const double *pose, const double *mix, bool with_odometer, State &state) {
    state.p = Eigen::Vector3d(pose[0], pose[1], pose[2]);
    state.q = Eigen::Quaterniond(pose[6], pose[3], pose[4], pose[5]);
    normlize(state.q);

    state.v  = Eigen::Vector3d(mix[0], mix[1], mix[2]);
    state.bg = Eigen::Vector3d(mix[3], mix[4], mix[5]);
    state.ba = Eigen::Vector3d(mix[6], mix[7], mix[8]);
    state.sodo = 0.0;   /* mix 里不再有 sodo 通道 */
}

void posePlus(const double *pose, const double *delta, double *pose_plus) {
    Eigen::Quaterniond q(pose[6], pose[3], pose[4], pose[5]);
    normlize(q);

    Eigen::Quaterniond q_plus = q * rv2q(Eigen::Vector3d(delta[3], delta[4], delta[5]));
    normlize(q_plus);

    pose_plus[0] = pose[0] + delta[0];
    pose_plus[1] = pose[1] + delta[1];
    pose_plus[2] = pose[2] + delta[2];
    pose_plus[3] = q_plus.x();
    pose_plus[4] = q_plus.y();
    pose_plus[5] = q_plus.z();
    pose_plus[6] = q_plus.w();
}

void poseMinus(const double *y, const double *x, double *y_minus_x) {
    Eigen::Quaterniond qy(y[6], y[3], y[4], y[5]);
    Eigen::Quaterniond qx(x[6], x[3], x[4], x[5]);
    normlize(qy);
    normlize(qx);

    //vect3 dphi = q2rv(qy * (~qx)); /*< qx -> qy 的旋转矢量 */
    Eigen::Vector3d dphi = q2rv(((qx.conjugate())) * qy);   /*< log(qx⁻¹ ⊗ qy)，与 posePlus 对偶 */

    y_minus_x[0] = y[0] - x[0];
    y_minus_x[1] = y[1] - x[1];
    y_minus_x[2] = y[2] - x[2];
    y_minus_x[3] = dphi.x();
    y_minus_x[4] = dphi.y();
    y_minus_x[5] = dphi.z();
}





/* 圆锥/划桨补偿系数表---------------------------------------*/
static const double conefactors[5][4] = {{2. / 3},
                                         {9. / 20, 27. / 20},
                                         {54. / 105, 92. / 105, 214. / 105},
                                         {250. / 504, 525. / 504, 650. / 504, 1375. / 504},
                                         {0.0, 0.0, 0.0, 0.0}};

IMU::IMU(void) {
    nSamples = 1;
    Reset();
    phim = dvbm = wmm = vmm = swmm = svmm = wm_1 = vm_1 = O31;
    tk  = 0.0;
    Kg  = Ka = I33;
    eb  = db = O31;
}

void IMU::Reset(void) {
    preFirst = onePlusPre = preWb = true;
    swmm = svmm = wm_1 = vm_1 = O31;
}

void IMU::SetKga(const Eigen::Matrix3d &Kg0, const Eigen::Vector3d eb0, const Eigen::Matrix3d &Ka0, const Eigen::Vector3d &db0) {
    Kg = Kg0;
    eb = eb0;
    Ka = Ka0;
    db = db0;
}

double IMU::Update(const Eigen::Vector3d *pwm, const Eigen::Vector3d *pvm, int nSamples0, double ts) {
    /* 不要用 assert 做这个边界检查：实际构建是 -O3 -DNDEBUG，assert 会被去掉，
     * 而 nSamples0==6 会让下面的循环读到 conefactors[4][4]（第 5 行只有 4 个
     * 元素）→ 越界读；nSamples0<=0 会用到未初始化的 pwm[0] 并让 nts=0。
     * 这里钳到受支持区间 [1, 5]（圆锥补偿最多 5 子样）。 */
    if (nSamples0 < 1) nSamples0 = 1;
    if (nSamples0 > 5) nSamples0 = 5;

    int i;
    const double *pcf = (nSamples0 >= 2) ? conefactors[nSamples0 - 2] : nullptr;
    Eigen::Vector3d cm = Eigen::Vector3d::Zero(), sm = Eigen::Vector3d::Zero();
    wmm = O31;
    vmm = O31;

    nSamples = nSamples0;
    nts      = static_cast<float>(nSamples0 * ts);
    _nts     = 1.0f / nts;
    tk += nts;

    if (nSamples0 == 1 && onePlusPre) { /* one-plus-previous sample */
        if (preFirst) {
            wm_1     = pwm[0];
            vm_1     = pvm[0];
            preFirst = false;
        }
        cm = wm_1 * (1.0 / 12.0);
        sm = vm_1 * (1.0 / 12.0);
    }

    for (i = 0; i < nSamples0 - 1; i++) {
        cm += pwm[i] * pcf[i];
        sm += pvm[i] * pcf[i];
        wmm += pwm[i];
        vmm += pvm[i];
    }
    wm_1 = pwm[i];
    vm_1 = pvm[i];
    wmm += pwm[i];
    swmm += wmm;
    vmm += pvm[i];
    svmm += vmm;

    phim = wmm + cm.cross(pwm[i]);
    dvbm = vmm + wmm.cross(vmm * 0.5) + (cm.cross(pvm[i]) + sm.cross(pwm[i]));

    return tk;
}


earth::earth(double a0, double f0) {
    a   = a0;
    f   = f0;
    wie = wie0;
    b   = (1.0 - f) * a;
    gn  = O31;
    pgn = nullptr;
    Update(O31, O31);
}

void earth::Init(double a0, double f0) {
    a   = a0;
    f   = f0;
    wie = wie0;
    b   = (1.0 - f) * a;
    gn  = O31;
    pgn = nullptr;
    Update(O31, O31);
}

void earth::Update(const Eigen::Vector3d &pos0, const Eigen::Vector3d &vn0) {
    pos = pos0;
    vn  = vn0;

    sl = std::sin(pos.x());
    cl = std::cos(pos.x());
    tl = sl / cl;

    double sq  = 1.0 - e2 * sl * sl;
    double sq2 = std::sqrt(sq);

    RMh    = a * (1.0 - e2) / sq / sq2 + pos.z();
    f_RMh  = 1.0 / RMh;
    RNh    = a / sq2 + pos.z();
    clRNh  = cl * RNh;
    f_RNh  = 1.0 / RNh;
    f_clRNh = 1.0 / clRNh;

    wnie = Eigen::Vector3d(0.0, wie * cl, wie * sl);
    wnen = Eigen::Vector3d(-vn.y() * f_RMh, vn.x() * f_RNh, vn.x() * f_RNh * tl);
    wnin = wnie + wnen;

    sl2 = sl * sl;
    sl4 = sl2 * sl2;
    gn  = Eigen::Vector3d(0.0, 0.0, -(G0 * (1.0 + 5.27094e-3 * sl2 + 2.32718e-5 * sl4) - 3.086e-6 * pos.z()));

    gcc = pgn ? *pgn : gn;
    gcc -= (wnie + wnin).cross(vn); /*< 叉乘：-(wie+wen) x vn */
}

Eigen::Vector3d earth::vn2dpos(const Eigen::Vector3d &vn0, float ts) const {
    return Eigen::Vector3d(vn0.y() * f_RMh, vn0.x() * f_clRNh, vn0.z()) * static_cast<double>(ts);
}

/* ============================================================================
 * Preintegration 实现
 * ==========================================================================*/
Preintegration::Preintegration(void) { reset(); }

Preintegration::Preintegration(const PreintegrationParam &param) {
    reset();
    param_ = param;
}

void Preintegration::reset(void) {
    dt_          = 0.0;
    num_         = 0;
    p_           = O31;
    v_           = O31;
    q_           = qI;
    bg_          = O31;
    ba_          = O31;
    last_dtheta_ = O31;
    last_dvel_   = O31;
    last_dt_     = 0.0;
    first_       = true;
    cov_.setZero();
    jac_.setIdentity();
    meas_list_.clear();

    /* 里程计预积分量 */
    s_     = O31;
    sodo_  = 0.0;
}

void Preintegration::setBias(const Eigen::Vector3d &bg, const Eigen::Vector3d &ba) {
    const double dbg = (bg - bg_).norm();
    const double dba = (ba - ba_).norm();
    bg_ = bg;
    ba_ = ba;
    if (dbg > 1.0e-10 || dba > 1.0e-10) {
        repropagation();
    }
}

void Preintegration::integration(const ImuMeas &meas) {
    if (meas.dt <= 1.0e-9) {
        return;
    }
    meas_list_.push_back(meas);
    propagate(meas);
}

void Preintegration::repropagation(void) {
    const Eigen::Vector3d bg_save = bg_;
    const Eigen::Vector3d ba_save = ba_;
    const std::deque<ImuMeas> meas_save = meas_list_;

    /* 清空累加状态（保留 bias 与测量队列） */
    dt_          = 0.0;
    num_         = 0;
    p_           = O31;
    v_           = O31;
    q_           = qI;
    last_dtheta_ = O31;
    last_dvel_   = O31;
    last_dt_     = 0.0;
    first_       = true;
    cov_.setZero();
    jac_.setIdentity();
    s_           = O31;
    sodo_        = 0.0;

    bg_ = bg_save;
    ba_ = ba_save;
    for (const auto &m : meas_save) {
        propagate(m);
    }
}

/* --------------------------------------------------------------------------
 * 单步中值积分
 *
 *   记区间起点体轴系为 b_k，终点为 b_{k+1}；Δq 表示"b_k -> b_0"的旋转
 *   （即 Δq * x_{b_k} = x_{b_0}）
 *
 *   去偏后测量：dθ̅ = dθ − bg·dt，d v̅ = dv − ba·dt
 *
 *   单步旋转：  Δq_{k+1} = Δq_k ⊗ Exp(dθ̅)
 *   中点旋转：  Δq_mid   = Δq_k ⊗ Exp(0.5·dθ̅)
 *   速度：      Δv_{k+1} = Δv_k + R(Δq_mid)·(dv̅/dt)·dt
 *   位置：      Δp_{k+1} = Δp_k + Δv_k·dt + 0.5·R(Δq_mid)·(dv̅/dt)·dt²
 *
 *   协方差按 Δx = [δp, δv, δφ, δbg, δba]（15 维）传播：
 *      δp'  = δp + δv·dt − 0.5·R·[a]×·δφ·dt² − 0.5·R·δba·dt²
 *      δv'  = δv − R·[a]×·δφ·dt − R·δba·dt
 *      δφ'  = δφ − [ω]×·δφ·dt − δbg·dt
 *      δbg' = δbg,  δba' = δba
 * -------------------------------------------------------------------------- */
void Preintegration::propagate(const ImuMeas &meas) {
    const double dt = meas.dt;
    if (dt <= 1.0e-9) {
        return;
    }

    /* 去偏 */
    const Eigen::Vector3d dtheta = meas.dtheta - bg_ * dt;
    const Eigen::Vector3d dvel   = meas.dvel   - ba_ * dt;

    const Eigen::Vector3d dtheta_e = dtheta;
    const Eigen::Vector3d dvel_e   = dvel;

    /* 单步旋转 & 半拍旋转 */
    Eigen::Quaterniond dq_step = rv2q(dtheta);
    normlize(dq_step);

    Eigen::Quaterniond dq_half = rv2q(dtheta * 0.5);
    normlize(dq_half);

    const Eigen::Quaterniond q_prev = q_;
    Eigen::Quaterniond q_mid = q_prev * dq_half;
    normlize(q_mid);

    Eigen::Matrix3d R_prev = q_prev.toRotationMatrix();
    Eigen::Matrix3d R_mid  = q_mid .toRotationMatrix();

    const Eigen::Vector3d a_body = dvel_e / dt;

    /* 位置/速度更新 */
    const Eigen::Vector3d v_old = v_;
    const Eigen::Vector3d a_b0 = R_mid * a_body;

    p_ = p_ + v_old * dt + Eigen::Vector3d(a_b0 * (0.5 * dt * dt));
    v_ = v_old + Eigen::Vector3d(a_b0 * dt);

    q_ = q_prev * dq_step;
    normlize(q_);

    /* 协方差传播 */
    const Eigen::Vector3d w_body = dtheta_e / dt;
    CovMatrix   F = buildF(a_body, w_body, R_prev, dt);
    GainMatrix  G = buildG(R_prev);
    NoiseMatrix Q = buildQ(dt);

    cov_ = F * cov_ * F.transpose() + G * Q * G.transpose();
    jac_ = F * jac_;

    if (meas.odovel != 0.0) {
        /* 安装角 C_b^m * e_x。abv = [pitch, 0, yaw] */
        const double ap = param_.abv.x();   /* pitch */
        const double ay = param_.abv.y();   /* yaw   */
        const Eigen::Vector3d cx(
             std::cos(ay) * std::cos(ap),
            -std::sin(ay) * std::cos(ap),
            -std::sin(ap));

        /* 刻度因子 */
        const double ds_scaled = (1.0 + sodo_) * meas.odovel;

        /* 杆臂补偿：omega_ib^b × l_OD；dtheta_e 是已经去偏后的角增量 */
        const Eigen::Vector3d w_body_b = dtheta_e / dt;
        const Eigen::Vector3d v_lev =  w_body_b.cross(param_.lvOD);

        const Eigen::Vector3d ds_body = ds_scaled * cx + v_lev * dt;
        s_ = s_ + Eigen::Vector3d(R_mid * ds_body);
    }

    /* 缓存 */
    last_dtheta_ = meas.dtheta;
    last_dvel_   = meas.dvel;
    last_dt_     = dt;
    dt_         += dt;
    num_++;

    first_ = false;   // 保留标记位供外部查询，但不再用于分支
}

/* --------------------------------------------------------------------------
 * F = I + A·dt（一阶离散），A 为 15×15 误差状态转移矩阵
 *   误差状态：δx = [δp, δv, δφ, δbg, δba]
 *   R 为区间起点的旋转（用于把体轴系加速度/角速度旋到 b0 系）
 * -------------------------------------------------------------------------- */
CovMatrix Preintegration::buildF(const Eigen::Vector3d &a_body,
                                                 const Eigen::Vector3d &w_body,
                                                 const Eigen::Matrix3d &R,
                                                 double dt) const {
    CovMatrix F = CovMatrix::Identity();

    Eigen::Matrix3d a_skew;
    a_skew <<             0.0, -a_body.z(),  a_body.y(),
                a_body.z(),             0.0, -a_body.x(),
               -a_body.y(),  a_body.x(),             0.0;

    Eigen::Matrix3d w_skew;
    w_skew <<             0.0, -w_body.z(),  w_body.y(),
                w_body.z(),             0.0, -w_body.x(),
               -w_body.y(),  w_body.x(),             0.0;

    /* δp' = δp + δv·dt − 0.5·R·[a]×·δφ·dt² − 0.5·R·δba·dt² */
    F.block<3, 3>(0, 3)  = Eigen::Matrix3d::Identity() * dt;
    F.block<3, 3>(0, 6)  = -0.5 * R * a_skew * dt * dt;
    F.block<3, 3>(0, 12) = -0.5 * R * dt * dt;

    /* δv' = δv − R·[a]×·δφ·dt − R·δba·dt */
    F.block<3, 3>(3, 6)  = -R * a_skew * dt;
    F.block<3, 3>(3, 12) = -R * dt;

    /* δφ' = δφ − [ω]×·δφ·dt − δbg·dt */
    F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() - w_skew * dt;
    F.block<3, 3>(6, 9) = -Eigen::Matrix3d::Identity() * dt;

    return F;
}

GainMatrix Preintegration::buildG(const Eigen::Matrix3d &R) const {
    GainMatrix G = GainMatrix::Zero();
    G.block<3, 3>(3, 3)  = -R;                            /* δv  ← na  */
    G.block<3, 3>(6, 0)  = -Eigen::Matrix3d::Identity();  /* δφ  ← ng  */
    G.block<3, 3>(9, 6)  =  Eigen::Matrix3d::Identity();  /* δbg ← nbg */
    G.block<3, 3>(12, 9) =  Eigen::Matrix3d::Identity();  /* δba ← nba */
    return G;
}

/* --------------------------------------------------------------------------
 * 过程噪声 Q（12×12，已含 dt 因子）
 *   白噪声：      Q_c = σ²，      离散 Q_d = σ²·dt
 *   零偏 GM 过程：Q_c = 2σ²/τ，   离散 Q_d = 2σ²/τ·dt
 *   σg, σa 分别来自 ARW / VRW；
 *   σbg, σba 是零偏稳态标准差；τ = corr_time。
 * -------------------------------------------------------------------------- */
NoiseMatrix Preintegration::buildQ(double dt) const {
    NoiseMatrix Q = NoiseMatrix::Zero();

    const double gyr_psd = param_.gyr_arw * param_.gyr_arw;
    const double acc_psd = param_.acc_vrw * param_.acc_vrw;

    const double tau    = std::max(param_.corr_time, 1.0); /*< 避免除零 */
    const double bg_psd = 2.0 * param_.gyr_bias_std * param_.gyr_bias_std / tau;
    const double ba_psd = 2.0 * param_.acc_bias_std * param_.acc_bias_std / tau;

    Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (gyr_psd * dt);
    Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (acc_psd * dt);
    Q.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (bg_psd  * dt);
    Q.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * (ba_psd  * dt);
    return Q;
}

State plus(const State &s, const DeltaN &delta) {
    State r = s;

    /* 块序与 README X.19 / buildF / buildG / buildQ / PreintResidual 一致：
     *   [δp(0..2), δv(3..5), δφ(6..8), δbg(9..11), δba(12..14)] */
    const Eigen::Vector3d dp  (delta(0),  delta(1),  delta(2) );
    const Eigen::Vector3d dv  (delta(3),  delta(4),  delta(5) );
    const Eigen::Vector3d dphi(delta(6),  delta(7),  delta(8) );
    const Eigen::Vector3d dbg (delta(9),  delta(10), delta(11));
    const Eigen::Vector3d dba (delta(12), delta(13), delta(14));

    /* 平移 / 速度 / 零偏：直接相加 */
    r.p  = s.p  + dp;
    r.v  = s.v  + dv;
    r.bg = s.bg + dbg;
    r.ba = s.ba + dba;

    /* 旋转：右扰动，扰动向量在体轴系 */
    r.q  = s.q * rv2q(dphi);
    normlize(r.q);    /* 防止长期迭代中数值漂移 */

    return r;
}

DeltaN minus(const State &a, const State &b) {
    DeltaN d;
    d.setZero();

    /* 平移 / 速度 / 零偏：直接相减 */
    d.segment<3>(0)  = (a.p  - b.p );
    d.segment<3>(3)  = (a.v  - b.v );   /* 与 plus / buildF 同为 [δp,δv,δφ,…] */
    d.segment<3>(9)  = (a.bg - b.bg);
    d.segment<3>(12) = (a.ba - b.ba);

    /* 旋转：δφ = Log( q_b⁻¹ ⊗ q_a )
     * 与 plus 使用的右扰动严格对偶：
     *   设 a.q = b.q ⊗ Exp(δφ)  ⟹  δφ = Log( b.q⁻¹ ⊗ a.q )
     */
    d.segment<3>(6)  = q2rv(b.q.conjugate() * a.q);

    return d;
}

void applyDelta(State &s, const DeltaN &delta) {
    s = plus(s, delta);
}

/* ============================================================
 * 16 维（带里程计比例因子 sodo）
 * ============================================================ */
State plus(const State &s, const DeltaNOdo &delta) {
    State r = s;

    /* 块序与 README X.19 / buildF / buildG / buildQ / PreintResidual 一致：
     *   [δp(0..2), δv(3..5), δφ(6..8), δbg(9..11), δba(12..14)] */
    const Eigen::Vector3d dp  (delta(0),  delta(1),  delta(2) );
    const Eigen::Vector3d dv  (delta(3),  delta(4),  delta(5) );
    const Eigen::Vector3d dphi(delta(6),  delta(7),  delta(8) );
    const Eigen::Vector3d dbg (delta(9),  delta(10), delta(11));
    const Eigen::Vector3d dba (delta(12), delta(13), delta(14));

    r.p  = s.p  + dp;
    r.v  = s.v  + dv;
    r.bg = s.bg + dbg;
    r.ba = s.ba + dba;
    r.q  = s.q * rv2q(dphi);
    normlize(r.q);

    r.sodo = s.sodo + delta(15);   /* 比例因子误差：直接加性 */

    return r;
}

DeltaNOdo minusWithOdo(const State &a, const State &b) {
    DeltaNOdo d;
    d.setZero();
    d.head<NUM_STATE>() = minus(a, b);   /* 复用 15 维实现 */
    d(15) = a.sodo - b.sodo;
    return d;
}

void applyDelta(State &s, const DeltaNOdo &delta) {
    s = plus(s, delta);
}

/* ============================================================
 * 数据桥接
 * ============================================================ */
void toData(const State &s, double *pose, double *mix, bool with_odometer) {
    /* pose = {pE, pN, pU, qx, qy, qz, qw} */
    pose[0] = s.p.x();
    pose[1] = s.p.y();
    pose[2] = s.p.z();
    pose[3] = s.q.x();   /* x */
    pose[4] = s.q.y();   /* y */
    pose[5] = s.q.z();   /* z */
    pose[6] = s.q.w();   /* w */

    /* mix = {v, bg, ba [, sodo]} */
    mix[0] = s.v.x();
    mix[1] = s.v.y();
    mix[2] = s.v.z();
    mix[3] = s.bg.x();
    mix[4] = s.bg.y();
    mix[5] = s.bg.z();
    mix[6] = s.ba.x();
    mix[7] = s.ba.y();
    mix[8] = s.ba.z();
    (void)with_odometer;   /* mix 现为 9 维；sodo 由 calib[0] 承载 */
}

State fromData(const double *pose, const double *mix, bool with_odometer) {
    State s;

    s.p = Eigen::Vector3d(pose[0], pose[1], pose[2]);

    /* 注意构造顺序：(w, x, y, z) */
    s.q = Eigen::Quaterniond(pose[6], pose[3], pose[4], pose[5]);
    normlize(s.q);

    s.v    = Eigen::Vector3d(mix[0], mix[1], mix[2]);
    s.bg   = Eigen::Vector3d(mix[3], mix[4], mix[5]);
    s.ba   = Eigen::Vector3d(mix[6], mix[7], mix[8]);
    s.sodo = with_odometer ? mix[9] : 0.0;

    return s;
}


/* ============================================================================
 * 法方程累积
 * ==========================================================================*/
void accumulate_normal_equations(const Eigen::MatrixXd &J, const Eigen::VectorXd &r,
                                 Eigen::MatrixXd &H, Eigen::VectorXd &b) {
    /* 形状检查必须显式做：实际构建是 -O3 -DNDEBUG，assert 会失效，
     * 而形状不符会让下面的 noalias() 越界读写。 */
    if (J.rows() != r.size()) {
        return;
    }
    if (H.size() == 0) {
        H = Eigen::MatrixXd::Zero(J.cols(), J.cols());
    }
    if (b.size() == 0) {
        b = Eigen::VectorXd::Zero(J.cols());
    }
    if (H.rows() != J.cols() || H.cols() != J.cols() || b.size() != J.cols()) {
        return;   /* 列布局漂移：宁可拒绝，也不要静默算错 */
    }

    /* H += Jᵀ J, b += Jᵀ r
     * 用 selfadjointView<Lower>().rankUpdate 更高效，但为了通用
     * （J 可能行数 > 列数，也可能列数 > 行数），直接用矩阵乘法。
     * 这里不再需要容器与 Eigen 之间的往返深拷贝。 */
    H.noalias() += J.transpose() * J;
    b.noalias() += J.transpose() * r;
}

/* ============================================================================
 * Schur 补
 * ==========================================================================*/
bool schur_complement(const Eigen::MatrixXd &H, const Eigen::VectorXd &b, int num_marginalized,
                      Eigen::MatrixXd &J_out, Eigen::VectorXd &r_out) {
    const int n = static_cast<int>(H.rows());
    if (H.rows() != H.cols() || b.size() != n || n == 0) {
        return false;
    }
    if (num_marginalized <= 0 || num_marginalized >= n) {
        return false;  /* 没有可边缘化的维，或会被边缘化光 */
    }
    const int m = num_marginalized;
    const int r = n - m;

    /* ---- 分块 ---- */
    Eigen::MatrixXd He = 0.5 * (H + H.transpose());   /* 强制对称，避免数值误差 */
    const Eigen::VectorXd &be = b;

    const Eigen::MatrixXd H_mm = He.topLeftCorner(m, m);
    const Eigen::MatrixXd H_mr = He.topRightCorner(m, r);
    const Eigen::MatrixXd H_rr = He.bottomRightCorner(r, r);
    const Eigen::VectorXd b_m  = be.head(m);
    const Eigen::VectorXd b_r  = be.tail(r);

    /* ---- 求解 H_mm·X = H_mr, H_mm·y = b_m ---- */
    Eigen::MatrixXd X;      /* = H_mm⁻¹ · H_mr */
    Eigen::VectorXd y;      /* = H_mm⁻¹ · b_m  */
    bool solved = false;

    Eigen::LDLT<Eigen::MatrixXd> ldlt(H_mm);
    if (ldlt.info() == Eigen::Success) {
        X = ldlt.solve(H_mr);
        y = ldlt.solve(b_m);
        solved = X.allFinite() && y.allFinite();
    }

    if (!solved) {
        /* 兜底：特征分解 + 相对阈值截断的伪逆
         * （H_mm 可能出现极小的负特征值，直接 LDLT/LLT 会失败） */
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_mm);
        if (es.info() != Eigen::Success) {
            return false;
        }
        const Eigen::VectorXd d = es.eigenvalues();
        const Eigen::MatrixXd V = es.eigenvectors();
        const double dmax = d.cwiseAbs().maxCoeff();
        if (!(dmax > 0.0) || !std::isfinite(dmax)) {
            return false;
        }
        const double thr = 1.0e-10 * dmax;
        Eigen::VectorXd dinv(d.size());
        for (int i = 0; i < d.size(); ++i) {
            dinv(i) = (std::fabs(d(i)) > thr) ? (1.0 / d(i)) : 0.0;
        }
        const Eigen::MatrixXd H_mm_inv = V * dinv.asDiagonal() * V.transpose();
        X = H_mm_inv * H_mr;
        y = H_mm_inv * b_m;
    }

    /* ---- 组装 H*, b* ---- */
    Eigen::MatrixXd H_star = H_rr - H_mr.transpose() * X;
    H_star = 0.5 * (H_star + H_star.transpose()).eval();
    const Eigen::VectorXd b_star = b_r - H_mr.transpose() * y;

    /* ---- 分解 H* = J_outᵀ J_out ---- */
    Eigen::MatrixXd J_out_e, r_out_e;

    const Eigen::MatrixXd L = llt_L(H_star);
    if (L.rows() > 0) {
        /* H_star = L·Lᵀ，取 J_out = Lᵀ（上三角）
         * J_outᵀ r_out = b*  ⟹  L·r_out = b*  ⟹  r_out = L⁻¹·b* */
        J_out_e = L.transpose();
        const Eigen::MatrixXd Le = L;
        r_out_e = Le.triangularView<Eigen::Lower>().solve(b_star);
    } else {
        /* 兜底：H_star = V·D·Vᵀ，负特征值钳到 0
         * 取 J_out = D^{1/2}·Vᵀ，则 J_outᵀ r_out = V·D^{1/2}·r_out = b*
         * 令 r_out = D^{-1/2}·Vᵀ·b*（对 D ≈ 0 的分量置 0） */
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_star);
        if (es.info() != Eigen::Success) {
            return false;
        }
        const Eigen::VectorXd d = es.eigenvalues();
        const Eigen::MatrixXd V = es.eigenvectors();
        const double dmax = d.cwiseAbs().maxCoeff();
        if (!(dmax > 0.0) || !std::isfinite(dmax)) {
            return false;
        }
        const double thr = 1.0e-12 * dmax;
        Eigen::VectorXd dsqrt (d.size());
        Eigen::VectorXd dinv_sq(d.size());
        for (int i = 0; i < d.size(); ++i) {
            if (d(i) > thr) {
                dsqrt(i)  = std::sqrt(d(i));
                dinv_sq(i) = 1.0 / dsqrt(i);
            } else {
                dsqrt(i)  = 0.0;
                dinv_sq(i) = 0.0;
            }
        }
        J_out_e = dsqrt.asDiagonal() * V.transpose();
        r_out_e = dinv_sq.asDiagonal() * (V.transpose() * b_star);
    }

    /* 返回前必须检查有限性：本函数的输出会被直接当作边缘化先验的
     * (J, r) 使用，一旦含 NaN/Inf 会静默污染整个优化问题。
     * （历史上 llt_L 的钳位分支在这里产出过 NaN。） */
    if (!J_out_e.allFinite() || !r_out_e.allFinite()) {
        J_out.resize(0, 0);
        r_out.resize(0);
        return false;
    }

    J_out = J_out_e;
    r_out = r_out_e;
    return true;
}

/* ============================================================================
 * 高层封装
 * ==========================================================================*/
void MarginalizationInfo::reset() {
    n_ = 0;
    H_.resize(0, 0);
    b_.resize(0);
    J_prior_.resize(0, 0);
    r_prior_.resize(0);
}

void MarginalizationInfo::addResidual(const Eigen::MatrixXd &J, const Eigen::VectorXd &r) {
    accumulate_normal_equations(J, r, H_, b_);
    n_ = static_cast<int>(H_.rows());
}

void MarginalizationInfo::setPrior(const Eigen::MatrixXd &J_out, const Eigen::VectorXd &r_out) {
    J_prior_ = J_out;
    r_prior_ = r_out;
    /* 先验法方程 H += J_outᵀ J_out, b += J_outᵀ r_out */
    accumulate_normal_equations(J_out, r_out, H_, b_);
    n_ = static_cast<int>(H_.rows());
}

bool MarginalizationInfo::marginalize(int num_marginalized) {
    if (n_ == 0) {
        return false;
    }
    Eigen::MatrixXd J_new;
    Eigen::VectorXd r_new;
    if (!schur_complement(H_, b_, num_marginalized, J_new, r_new)) {
        return false;
    }
    /* 边缘化完成后，把结果保存为先验，并让累积状态回到"只有先验"的状态 */
    J_prior_ = J_new;
    r_prior_ = r_new;
    H_.resize(0, 0);
    b_.resize(0);
    accumulate_normal_equations(J_prior_, r_prior_, H_, b_);
    n_ = static_cast<int>(H_.rows());
    return true;
}

/* ============================================================================
 * VisualFrontend 实现
 * ==========================================================================*/
VisualFrontend::VisualFrontend(const CameraModel& cam, int max_feat, int min_dist)
    : cam_(cam), max_feat_(max_feat), min_dist_(min_dist) {}

std::vector<TrackedFeature> VisualFrontend::process(double /*t*/, const cv::Mat& gray) {
    std::vector<TrackedFeature> result;
    std::vector<cv::Point2f>    cur_pts;
    std::vector<int>            cur_ids, cur_cnt;

    if (first_frame_) {
        cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(255));
        cv::goodFeaturesToTrack(gray, cur_pts, max_feat_, 0.01, min_dist_, mask);
        for (size_t i = 0; i < cur_pts.size(); ++i) {
            cur_ids.push_back(next_id_++);
            cur_cnt.push_back(1);
        }
        first_frame_ = false;
    } else {
        /* LK 前向追踪 */
        std::vector<uchar> status; std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_img_, gray, prev_pts_, cur_pts,
                                 status, err, cv::Size(21,21), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT +
                                                  cv::TermCriteria::EPS, 30, 0.01),
                                 0);

        /* 反向追踪做前后向校验 */
        std::vector<cv::Point2f> back_pts;
        std::vector<uchar> back_status; std::vector<float> back_err;
        cv::calcOpticalFlowPyrLK(gray, prev_img_, cur_pts, back_pts,
                                 back_status, back_err, cv::Size(21,21), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT +
                                                  cv::TermCriteria::EPS, 30, 0.01), 0);

        for (size_t i = 0; i < cur_pts.size(); ++i) {
            if (!status[i]) continue;
            const double dx = back_pts[i].x - prev_pts_[i].x;
            const double dy = back_pts[i].y - prev_pts_[i].y;
            if (dx*dx + dy*dy > 0.25) status[i] = 0;   /* 0.5 px 阈值 */
        }

        /* RANSAC F 矩阵剔除 */
        rejectWithF(prev_pts_, cur_pts, status);

        std::vector<cv::Point2f> kept_pts;
        for (size_t i = 0; i < cur_pts.size(); ++i) {
            if (!status[i]) continue;
            kept_pts.push_back(cur_pts[i]);
            cur_ids.push_back(prev_ids_[i]);
            cur_cnt.push_back(track_cnt_[i] + 1);
        }

        /* 用 mask 补足新点 */
        cv::Mat mask = makeMask(kept_pts);
        const int need = max_feat_ - static_cast<int>(kept_pts.size());
        if (need > 0) {
            std::vector<cv::Point2f> new_pts;
            cv::goodFeaturesToTrack(gray, new_pts, need, 0.01, min_dist_, mask);
            for (const auto& p : new_pts) {
                kept_pts.push_back(p);
                cur_ids.push_back(next_id_++);
                cur_cnt.push_back(1);
            }
        }
        cur_pts = std::move(kept_pts);
    }

    result.reserve(cur_pts.size());
    for (size_t i = 0; i < cur_pts.size(); ++i) {
        TrackedFeature tf;
        tf.feat_id   = cur_ids[i];
        tf.pt        = cur_pts[i];
        tf.xyz_c     = undistortToNorm(cur_pts[i]);
        tf.track_cnt = cur_cnt[i];
        result.push_back(tf);
    }

    prev_img_  = gray.clone();
    prev_pts_  = std::move(cur_pts);
    prev_ids_  = std::move(cur_ids);
    track_cnt_ = std::move(cur_cnt);
    return result;
}

cv::Mat VisualFrontend::makeMask(const std::vector<cv::Point2f>& pts) const {
    cv::Mat mask(prev_img_.size(), CV_8UC1, cv::Scalar(255));
    for (const auto& p : pts) cv::circle(mask, p, min_dist_, cv::Scalar(0), -1);
    return mask;
}

void VisualFrontend::rejectWithF(const std::vector<cv::Point2f>& p0,
                                 const std::vector<cv::Point2f>& p1,
                                 std::vector<uchar>& status) {
    if (p0.size() < 8) return;
    std::vector<cv::Point2f> p0_in, p1_in; std::vector<int> idx;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) { p0_in.push_back(p0[i]); p1_in.push_back(p1[i]);
                         idx.push_back((int)i); }
    }
    if (p0_in.size() < 8) return;

    std::vector<uchar> fmask;
    cv::findFundamentalMat(p0_in, p1_in, cv::FM_RANSAC, 1.0, 0.99, fmask);
    if (fmask.empty()) return;   /* F 矩阵退化，放弃剔除 */

    /* 统计内点数：太少说明几何退化（纯平移/纯旋转），F 不可信 */
    int inliers = 0;
    for (uchar v : fmask) if (v) inliers++;
    if (inliers < 8) return;

    for (size_t k = 0; k < fmask.size(); ++k)
        if (!fmask[k]) status[idx[k]] = 0;
}

Eigen::Vector3d VisualFrontend::undistortToNorm(const cv::Point2f& pt) const {
    std::vector<cv::Point2f> in{pt}, out;
    cv::undistortPoints(in, out, cam_.K(), cam_.D());
    return Eigen::Vector3d(out[0].x, out[0].y, 1.0);
}

/* ============================================================================
 * 离线预处理：把图像目录跑一遍前端，生成时间戳 -> VisualObs 表
 * ==========================================================================*/
std::map<double, std::vector<VisualObs>>
preprocessImages(const std::string& cam_dir,
                 const CameraModel& cam,
                 double visual_sigma,
                 int    max_feat,
                 int    min_dist) {
    namespace fs = std::filesystem;
    std::map<double, std::vector<VisualObs>> table;
    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(cam_dir)) {
        auto ext = e.path().extension().string();
        if (ext == ".png" || ext == ".jpg") files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::fprintf(stderr, "[visual] %s 下没有图像\n", cam_dir.c_str());
        return table;
    }

    VisualFrontend frontend(cam, max_feat, min_dist);
    for (const auto& f : files) {
        const std::string stem = fs::path(f).stem().string();
        double t_ns = 0.0;
        try { t_ns = std::stod(stem); } catch (...) { continue; }
        const double t = t_ns * 1e-9;   /* 纳秒 -> 秒 */

        cv::Mat img = cv::imread(f, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        auto feats = frontend.process(t, img);

        std::vector<VisualObs> obs;
        obs.reserve(feats.size());
        for (const auto& tf : feats) {
            if (tf.track_cnt < 2) continue;   /* 只保留稳定追踪点 */
            VisualObs vo;
            vo.feat_id = tf.feat_id;
            vo.xyz_c   = Eigen::Vector3d(tf.xyz_c(0), tf.xyz_c(1), 1.0);
            vo.sigma   = visual_sigma;
            obs.push_back(vo);
        }
        table[t] = std::move(obs);
    }
    std::printf("[visual] 预处理 %zu 帧图像, 表大小 %zu\n",
                files.size(), table.size());
    return table;
}

/* ============================================================================
 * GraphOptimizer::buildProblem
 * ==========================================================================*/
void GraphOptimizer::buildProblem(ceres::Problem &problem,
                                  std::vector<Frame> &frames,
                                  const Eigen::Vector3d &g_n,
                                  std::vector<double*> *ordered) {
    /* 记录参数块的加入顺序。EvaluateOptions.parameter_blocks 用这个顺序
     * 决定 CRS 雅可比的列序 —— 边缘化就是靠它把"要消去的列"放到最前面。 */
    auto addBlock = [&](double *p, int size) {
        problem.AddParameterBlock(p, size);
        if (ordered) ordered->push_back(p);
    };

    /* ---- 0) 标定参数块（全局，窗口内共享） ---- */
    addBlock(calib_data_.data(), NUM_CALIB);
    problem.SetManifold(calib_data_.data(), new CalibManifold());
    if (!opt_.calib_mode) {
        problem.SetParameterBlockConstant(calib_data_.data());   /* 导航模式：固定 */
    }

    /* ---- 1) 帧参数块 + Manifold ---- */
    for (auto &f : frames) {
        addBlock(f.pose.data(), NUM_POSE);
        problem.SetManifold(f.pose.data(), new PoseManifold());
        addBlock(f.mix.data(), NUM_MIX_ODO);
    }
    if (opt_.fix_first_pose && !frames.empty()) {
        problem.SetParameterBlockConstant(frames.front().pose.data());
    }

    /* ---- 2) 预积分因子 ---- */
    for (size_t k = 0; k + 1 < frames.size(); ++k) {
        if (!frames[k].pre_to_next) continue;
        auto *cost = new ceres::AutoDiffCostFunction<PreintResidual,
                                                     15, 7, NUM_MIX_ODO, 7, NUM_MIX_ODO>(
            new PreintResidual(*frames[k].pre_to_next, g_n,
                               opt_.whiten_preint, opt_.bias_jac));
        problem.AddResidualBlock(cost, nullptr,
                                 frames[k].pose.data(),   frames[k].mix.data(),
                                 frames[k+1].pose.data(), frames[k+1].mix.data());
    }

    /* ---- 3) GNSS 位置因子 ---- */
    for (auto &f : frames) {
        if (!f.has_gnss) continue;
        auto *cost = new ceres::AutoDiffCostFunction<GnssPosResidual, 3, 7>(
            new GnssPosResidual(f.gnss_pos, f.gnss_std));
        problem.AddResidualBlock(cost, nullptr, f.pose.data());
    }

    /* ---- 4) GNSS 速度因子 ---- */
    for (auto &f : frames) {
        if (!f.has_gnss_vel) continue;
        auto *cost = new ceres::AutoDiffCostFunction<GnssVelResidual, 3, NUM_MIX_ODO>(
            new GnssVelResidual(f.gnss_vn, f.gnss_vn_std, 1.5));
        problem.AddResidualBlock(cost, nullptr, f.mix.data());
    }

    /* ---- 5) GNSS 航向因子（引用标定块 calib[6]） ---- */
    for (auto &f : frames) {
        if (!f.has_gnss_yaw || f.gnss_yaw_std <= 0.0) continue;
        auto *cost = new ceres::AutoDiffCostFunction<GnssYawResidual, 1, 7, NUM_CALIB>(
            new GnssYawResidual(f.gnss_yaw, f.gnss_yaw_std, 1.5));
        problem.AddResidualBlock(cost, nullptr,
                                 f.pose.data(), calib_data_.data());
    }

    /* ---- 6) 静止零速因子 ---- */
    for (auto &f : frames) {
        if (!f.is_static) continue;
        auto *cost = new ceres::AutoDiffCostFunction<StaticVelResidual, 3, NUM_MIX_ODO>(
            new StaticVelResidual(opt_.static_vel_sigma, 1.0));
        problem.AddResidualBlock(cost, nullptr, f.mix.data());
    }

    /* ---- 7) 里程计速度因子（引用整个标定块） ---- */
    for (auto &f : frames) {
        if (!f.has_odo) continue;
        auto *cost = new ceres::AutoDiffCostFunction<OdoVelResidual, 3, 7, NUM_MIX_ODO, NUM_CALIB>(
            new OdoVelResidual(f.odo_dS, f.odo_dt, f.odo_omega_meas,
                               f.odo_std, 1.5));
        problem.AddResidualBlock(cost, nullptr,
                                 f.pose.data(), f.mix.data(),
                                 calib_data_.data());
    }

    if (opt_.use_visual) {
        std::unordered_map<int, std::vector<std::pair<int, const VisualObs*>>> obs_map;
        for (int k = 0; k < (int)frames.size(); ++k)
            for (const auto& vo : frames[k].visual_obs)
                obs_map[vo.feat_id].emplace_back(k, &vo);

        for (auto& kv : obs_map) {
            const int fid = kv.first;
            auto& obs_list = kv.second;
            if (obs_list.size() < 2) continue;

            auto it = inv_depths_.find(fid);
            if (it == inv_depths_.end()) { inv_depths_[fid] = 0.1; it = inv_depths_.find(fid); }
            double* inv_depth = &(it->second);

            addBlock(inv_depth, 1);
            problem.SetParameterLowerBound(inv_depth, 0, opt_.inv_depth_min);
            problem.SetParameterUpperBound(inv_depth, 0, opt_.inv_depth_max);

            const int        ai = obs_list.front().first;
            const VisualObs* a  = obs_list.front().second;
            for (size_t m = 1; m < obs_list.size(); ++m) {
                const int        ci = obs_list[m].first;
                const VisualObs* c  = obs_list[m].second;
                const double     sigma = std::max(a->sigma, c->sigma);

                /* 注意：不再用 AutoDiffCostFunction 包装，直接 new VisualReprojResidual */
                auto* cost = new VisualReprojResidual(a->xyz_c, c->xyz_c, sigma);
                auto* loss = new ceres::HuberLoss(opt_.visual_huber);
                problem.AddResidualBlock(
                    cost, loss,
                    frames[ai].pose.data(),
                    frames[ci].pose.data(),
                    calib_data_.data(),
                    inv_depth);
            }
        }
    }

    /* ============================================================================
    * GNSS 紧组合因子（多频）
    * ==========================================================================*/
    if (opt_.use_gnss_dd) {
        const int nf = std::min(std::max(opt_.max_freq, 1), MAX_FREQ);

        /* base_pos_ecef 必须显式设置：默认是全 0（地心）。忘了设的话
         * ‖b_i‖ 会变成整段站星距，双差完全不成立 —— 直接拒绝并说明原因。 */
        if (base_pos_v3_.norm() < 1.0) {
            if (!amb_warned_basepos_) {
                std::fprintf(stderr,
                    "[GNSS-DD] 基准站坐标无效（||base_pos_ecef||=%.3f，看起来仍是默认地心）。"
                    "双差需要 ECEF 下的基准站位置，请用 setBasePosEcef()/--base-pos 设置；"
                    "本次跳过全部 GNSS 双差因子。\n", base_pos_v3_.norm());
                amb_warned_basepos_ = true;
            }
            return;
        }

        int n_dd_frames = 0;
        for (const auto &fr : frames) if (fr.has_gnss_raw) ++n_dd_frames;
        if (n_dd_frames == 0) {
            if (!amb_warned_nodata_) {
                std::fprintf(stderr,
                    "[GNSS-DD] 已请求双差紧组合，但没有任何帧带 dd_obs（has_gnss_raw 全为 false）。"
                    "该模式目前没有数据生产者，紧组合因子不会生效，请不要当作已启用。\n");
                amb_warned_nodata_ = true;
            }
            return;
        }

        amb_used_keys_.clear();

        for (size_t k = 0; k < frames.size(); ++k) {
            auto& f = frames[k];
            if (!f.has_gnss_raw) continue;

            addBlock(f.clk_bias.data(),  1);
            addBlock(f.clk_drift.data(), 1);

            /* 模糊度容器
             *
             * 【重要】buildProblem 只允许**只读** amb_values / amb_fixed：
             * 整周固定信息由外部（整周搜索成功后）写回，这里清空会让
             * `opt_.fix_ambiguity && f.amb_fixed[j][fi]` 恒为 false，使
             * --rtk-fix 变成空开关；即使侥幸可达，amb_values 也会是 0，
             * 把模糊度硬钉到 0。因此这里只保证容器尺寸，不清零已有取值。 */
            f.amb_values.resize(f.dd_obs.size());
            f.amb_fixed .resize(f.dd_obs.size());
            f.amb_wl_values.resize(f.dd_obs.size(), 0.0);
            f.amb_wl_fixed .resize(f.dd_obs.size(), false);
            f.amb_if_values.resize(f.dd_obs.size(), 0.0);
            f.amb_if_fixed .resize(f.dd_obs.size(), false);

            for (size_t j = 0; j < f.dd_obs.size(); ++j) {
                const auto& dd = f.dd_obs[j];
                /* 不再 fill(0.0)/fill(false)：见上面的说明。 */

                /* ---- 逐频：伪距 + 载波 ---- */
                for (int fi = 0; fi < nf; ++fi) {
                    if (!dd.has_P[fi] && !dd.has_L[fi]) continue;

                    if (dd.has_P[fi]) {
                        auto* cost = new GnssDDPseudorangeFreqResidual(
                            dd, fi, base_pos_v3_, R_en_m3_, p0_ecef_v3_,
                            opt_.gnss_ant_lever);
                        problem.AddResidualBlock(cost, nullptr, f.pose.data());
                    }

                    if (dd.has_L[fi]) {
                        /* 弧段号：帧内该 DD 在该频点上的弧号。超出 dd_arc 范围时退化为 0
                        * （例如手工构造的测试帧没填 dd_arc）。 */
                        const int arc = (j < f.dd_arc.size() && fi < MAX_FREQ)
                                        ? f.dd_arc[j][fi] : 0;
                        const AmbKey key = ambKey(dd.ref_sat, dd.sat, fi, arc);
                        auto it = ambiguities_.find(key);
                        if (it == ambiguities_.end()) {
                            ambiguities_[key] = {0.0};
                            it = ambiguities_.find(key);
                        }
                        amb_used_keys_.insert(key);
                        double* amb = it->second.data();
                        addBlock(amb, 1);
                        problem.SetParameterLowerBound(amb, 0, -1.0e4);
                        problem.SetParameterUpperBound(amb, 0,  1.0e4);

                        auto* cost = new GnssDDCarrierFreqResidual(
                            dd, fi, base_pos_v3_, R_en_m3_, p0_ecef_v3_,
                            opt_.gnss_ant_lever);
                        problem.AddResidualBlock(cost, nullptr,
                                                f.pose.data(), amb);

                        if (opt_.fix_ambiguity && f.amb_fixed[j][fi]) {
                            auto* fix = new AmbiguityFixFactor(
                                std::round(f.amb_values[j][fi]), 1.0e-3);
                            problem.AddResidualBlock(fix, nullptr, amb);
                        }
                    }
                }

                /* ---- WL / NL ---- */
                if (opt_.use_wl_nl && nf >= 2 &&
                    dd.has_L[0] && dd.has_L[1]) {

                    const int arc_cmb = (j < f.dd_arc.size())? std::max(f.dd_arc[j][0], f.dd_arc[j][1]) : 0;
                    const AmbKey key_wl = ambKey(dd.ref_sat, dd.sat, 100, arc_cmb);
                    auto it_wl = ambiguities_.find(key_wl);
                    if (it_wl == ambiguities_.end()) {
                        ambiguities_[key_wl] = {0.0};
                        it_wl = ambiguities_.find(key_wl);
                    }
                    amb_used_keys_.insert(key_wl);
                    double* amb_wl = it_wl->second.data();
                    addBlock(amb_wl, 1);

                    auto* wl = new GnssDDWideLaneResidual(
                        dd, base_pos_v3_, R_en_m3_, p0_ecef_v3_,
                        opt_.gnss_ant_lever, opt_.wl_sigma);
                    problem.AddResidualBlock(wl, nullptr,
                                            f.pose.data(), amb_wl);

                    if (opt_.fix_ambiguity && f.amb_wl_fixed[j]) {
                        auto* fix = new AmbiguityFixFactor(
                            std::round(f.amb_wl_values[j]), 1.0e-3);
                        problem.AddResidualBlock(fix, nullptr, amb_wl);
                    }

                    const AmbKey key_nl = ambKey(dd.ref_sat, dd.sat, 200, arc_cmb);
                    auto it_nl = ambiguities_.find(key_nl);
                    if (it_nl == ambiguities_.end()) {
                        ambiguities_[key_nl] = {0.0};
                        it_nl = ambiguities_.find(key_nl);
                    }
                    amb_used_keys_.insert(key_nl);
                    double* amb_nl = it_nl->second.data();
                    addBlock(amb_nl, 1);

                    auto* nl = new GnssDDNarrowLaneResidual(
                        dd, base_pos_v3_, R_en_m3_, p0_ecef_v3_,
                        opt_.gnss_ant_lever, opt_.nl_sigma);
                    problem.AddResidualBlock(nl, nullptr,
                                            f.pose.data(), amb_nl);
                }

                /* ---- IF ---- */
                if (opt_.use_if_combination && nf >= 2 &&
                    dd.has_L[0] && dd.has_L[1]) {

                    const int arc_cmb = (j < f.dd_arc.size()) ? std::max(f.dd_arc[j][0], f.dd_arc[j][1]) : 0;
                    const AmbKey key_if = ambKey(dd.ref_sat, dd.sat, 300, arc_cmb);
                    auto it_if = ambiguities_.find(key_if);
                    if (it_if == ambiguities_.end()) {
                        ambiguities_[key_if] = {0.0};
                        it_if = ambiguities_.find(key_if);
                    }
                    amb_used_keys_.insert(key_if);
                    double* amb_if = it_if->second.data();
                    addBlock(amb_if, 1);

                    auto* iff = new GnssDDIFCarrierResidual(
                        dd, base_pos_v3_, R_en_m3_, p0_ecef_v3_,
                        opt_.gnss_ant_lever, opt_.if_sigma);
                    problem.AddResidualBlock(iff, nullptr,
                                            f.pose.data(), amb_if);
                }
            }

            /* 钟差 / 钟漂 RW（相邻帧） */
            if (k + 1 < frames.size() && frames[k + 1].has_gnss_raw) {
                const double dt = frames[k + 1].time - f.time;
                if (dt > 0.0) {
                    auto* rw = new ClockBiasRWFactor(dt, opt_.clk_rw_sigma);
                    problem.AddResidualBlock(rw, nullptr,
                                            f.clk_bias.data(),
                                            f.clk_drift.data(),
                                            frames[k + 1].clk_bias.data());

                    auto* dw = new ClockDriftRWFactor(opt_.clk_drift_rw_sigma);
                    problem.AddResidualBlock(dw, nullptr,
                                            f.clk_drift.data(),
                                            frames[k + 1].clk_drift.data());
                }
            }
        }
    
        /* 清理本窗口未再出现的旧键：键集只增不减会污染协方差提取 */
        ambPruneUnused(ambiguities_, amb_used_keys_);
    }
    
}

/* ============================================================================
 * 模糊度管理
 * ==========================================================================*/
int ambPromoteReference(AmbMap &m, int old_ref, int new_ref, int slot, int arc) {
    if (old_ref == new_ref) return 0;
    /* N_k^{k'}：旧参考 k 与新参考 k' 之间的模糊度 */
    auto it_kk = m.find(ambKey(old_ref, new_ref, slot, arc));
    const bool has_ref_diff = (it_kk != m.end());
    const double n_kk = has_ref_diff ? it_kk->second[0] : 0.0;

    int n_ok = 0;
    std::vector<std::pair<AmbKey, double>> add;
    for (auto &kv : m) {
        if (ambKeyRefSat(kv.first) != old_ref) continue;
        if (ambKeySlot(kv.first) != slot) continue;
        if (ambKeyArc(kv.first) != arc) continue;
        const int sat = ambKeySat(kv.first);
        if (sat == new_ref) continue;                 /* 新参考星自身没有 DD 模糊度 */
        if (!has_ref_diff) continue;                  /* 无法转换：留给新弧重新收敛 */
        add.emplace_back(ambKey(new_ref, sat, slot, arc), kv.second[0] - n_kk);
    }
    for (auto &p : add) m[p.first] = {p.second};
    n_ok = static_cast<int>(add.size());
    return n_ok;
}

AmbKey ambAdvanceArc(AmbMap &m, int ref_sat, int sat, int slot, int new_arc,
                     double slip_cycles) {
    const AmbKey new_key = ambKey(ref_sat, sat, slot, new_arc);
    if (m.find(new_key) == m.end()) {
        /* 以旧值（若在）为初值，按周跳量修正 */
        double v = 0.0;
        for (auto &kv : m) {
            if (ambKeyRefSat(kv.first) == ref_sat && ambKeySat(kv.first) == sat &&
                ambKeySlot(kv.first) == slot) {
                v = kv.second[0];
                break;
            }
        }
        m[new_key] = {v + slip_cycles};
    }
    return new_key;
}

void ambPruneUnused(AmbMap &m, const std::set<AmbKey> &used) {
    for (auto it = m.begin(); it != m.end();) {
        if (used.find(it->first) == used.end()) it = m.erase(it);
        else ++it;
    }
}

bool GraphOptimizer::extractAmbiguityCovariance(ceres::Problem& problem,
                                                Eigen::MatrixXd& Q_NN) {
    std::vector<double*> amb_ptrs;
    for (auto& [key, arr] : ambiguities_) amb_ptrs.push_back(arr.data());
    const int n = static_cast<int>(amb_ptrs.size());
    if (n == 0) return false;

    ceres::Covariance::Options cov_opt;
    cov_opt.algorithm_type  = ceres::DENSE_SVD;
    cov_opt.null_space_rank = -1;
    ceres::Covariance covariance(cov_opt);

    std::vector<std::pair<const double*, const double*>> blocks;
    for (auto* p : amb_ptrs) blocks.push_back({p, p});
    if (!covariance.Compute(blocks, &problem)) return false;

    Q_NN.resize(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            covariance.GetCovarianceBlock(amb_ptrs[i], amb_ptrs[j], &Q_NN(i, j));
    return true;
}

/* ============================================================================
 * GraphOptimizer：先验指针解析
 *
 * 先验里存的是"块身份"而不是裸指针 —— std::vector<Frame> 会 reallocate，
 * 滑窗每步还会 erase，指针会失效。这里按身份重新解析成当前参数块的指针。
 * ==========================================================================*/
bool GraphOptimizer::resolvePriorPointers(const std::vector<Frame> &frames,
                                          std::vector<double*> &out) {
    out.clear();
    if (!prior_) return false;
    out.reserve(prior_->refs.size());
    for (const PriorBlockRef &ref : prior_->refs) {
        switch (ref.kind) {
        case PriorRefKind::CALIB:
            out.push_back(calib_data_.data());
            break;
        case PriorRefKind::FRAME_POSE: {
            double *p = nullptr;
            for (const Frame &f : frames) {
                if (f.time == ref.time) { p = const_cast<double*>(f.pose.data()); break; }
            }
            if (!p) return false;
            out.push_back(p);
            break;
        }
        case PriorRefKind::FRAME_MIX: {
            double *p = nullptr;
            for (const Frame &f : frames) {
                if (f.time == ref.time) { p = const_cast<double*>(f.mix.data()); break; }
            }
            if (!p) return false;
            out.push_back(p);
            break;
        }
        case PriorRefKind::AMB: {
            auto it = ambiguities_.find(ref.amb);
            if (it == ambiguities_.end()) return false;
            out.push_back(it->second.data());
            break;
        }
        }
    }
    return out.size() == prior_->refs.size();
}

/* ============================================================================
 * GraphOptimizer::marginalizeOldestFrame
 *
 * 做法（与 VINS 系列一致的"部分边缘化"）：
 *   1. 只取**触及 frames[0]** 的残差块（外加已有的旧先验）；
 *      只涉及保留帧的因子不能进 Schur 补 —— 它们出窗后仍在问题里，
 *      进补就会把信息重复计一次（过度自信）。
 *   2. 用 EvaluateOptions.parameter_blocks 显式定列序，把 frames[0] 的
 *      参数块放在最前面，于是"被消去的切空间列"恰好是连续的前缀。
 *   3. 用 Ceres 报出的切空间维数（ParameterBlockTangentSize）算 num_marg，
 *      不再硬编码 16 —— calib 是否自由、fix_first_pose 是否打开都会影响它。
 *   4. Schur 补后把结果存成按块描述的先验 prior_。
 * ==========================================================================*/
bool GraphOptimizer::marginalizeOldestFrame(std::vector<Frame> &frames,
                                            const Eigen::Vector3d &g_n) {
    if (frames.size() < 2) return false;

    ceres::Problem problem;
    std::vector<double*> blocks;
    buildProblem(problem, frames, g_n, &blocks);
    if (blocks.empty()) return false;

    /* 旧先验纳入本次边缘化，信息才能沿窗口链式传递 */
    ceres::ResidualBlockId prior_id = nullptr;
    if (prior_ && prior_->J.rows() > 0) {
        std::vector<double*> ptrs;
        if (!resolvePriorPointers(frames, ptrs)) return false;
        prior_id = problem.AddResidualBlock(
            new MarginalizationPriorFactor(prior_->J, prior_->r, prior_->meta), nullptr, ptrs);
    }

    /* ---- 1) 要消去的块：frames[0] 的 pose / mix（非常量者） ---- */
    std::vector<double*> head;
    if (!problem.IsParameterBlockConstant(frames[0].pose.data())) head.push_back(frames[0].pose.data());
    if (!problem.IsParameterBlockConstant(frames[0].mix.data()))  head.push_back(frames[0].mix.data());
    if (head.empty()) return false;

    /* ---- 2) 显式列序：head 在最前，其余按 buildProblem 的固定顺序 ---- */
    ceres::Problem::EvaluateOptions eo;
    eo.apply_loss_function = false;
    eo.parameter_blocks = head;
    for (double *p : blocks) {
        if (std::find(head.begin(), head.end(), p) != head.end()) continue;
        if (problem.IsParameterBlockConstant(p)) continue;   /* 常量块不产生列 */
        eo.parameter_blocks.push_back(p);
    }

    /* ---- 3) 只取触及 frames[0] 的残差块 ---- */
    std::vector<ceres::ResidualBlockId> res_ids;
    for (double *p : head) {
        std::vector<ceres::ResidualBlockId> rb;
        problem.GetResidualBlocksForParameterBlock(p, &rb);
        for (auto id : rb) {
            if (std::find(res_ids.begin(), res_ids.end(), id) == res_ids.end()) res_ids.push_back(id);
        }
    }
    if (prior_id) res_ids.push_back(prior_id);
    if (res_ids.empty()) return false;
    eo.residual_blocks = res_ids;

    /* ---- 4) 求 J / r ---- */
    double cost = 0.0;
    std::vector<double> residuals, gradient;
    ceres::CRSMatrix jac;
    if (!problem.Evaluate(eo, &cost, &residuals, &gradient, &jac)) return false;

    const int nr = jac.num_rows;
    const int nc = jac.num_cols;
    if (nr <= 0 || nc <= 0) return false;
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(nr, nc);
    for (int row = 0; row < nr; ++row) {
        for (int k = jac.rows[row]; k < jac.rows[row + 1]; ++k) {
            J(row, jac.cols[k]) = jac.values[k];
        }
    }
    Eigen::VectorXd r(residuals.size());
    for (size_t k = 0; k < residuals.size(); ++k) r(k) = residuals[k];

    /* ---- 5) 法方程 + Schur 补 ---- */
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    accumulate_normal_equations(J, r, H, b);
    if (H.rows() != nc) return false;

    int num_marg = 0;
    for (double *p : head) num_marg += problem.ParameterBlockTangentSize(p);
    if (num_marg <= 0 || num_marg >= nc) return false;

    Eigen::MatrixXd J_out;
    Eigen::VectorXd r_out;
    if (!schur_complement(H, b, num_marg, J_out, r_out)) return false;

    /* ---- 6) 组装按块描述的新先验 ---- */
    auto spec = std::make_shared<PriorSpec>();
    spec->J = J_out;
    spec->r = r_out;

    /* 指针 -> 块身份 的反查表 */
    std::map<const double*, PriorBlockRef> ref_of;
    ref_of[calib_data_.data()] = PriorBlockRef{PriorRefKind::CALIB, 0.0, 0};
    for (auto &fr : frames) {
        ref_of[fr.pose.data()] = PriorBlockRef{PriorRefKind::FRAME_POSE, fr.time, 0};
        ref_of[fr.mix.data()]  = PriorBlockRef{PriorRefKind::FRAME_MIX,  fr.time, 0};
    }
    for (auto &kv : ambiguities_) {
        ref_of[kv.second.data()] = PriorBlockRef{PriorRefKind::AMB, 0.0, kv.first};
    }

    for (size_t k = head.size(); k < eo.parameter_blocks.size(); ++k) {
        double *p = eo.parameter_blocks[k];
        if (problem.IsParameterBlockConstant(p)) continue;
        const int amb = problem.ParameterBlockSize(p);
        const int tan = problem.ParameterBlockTangentSize(p);
        if (tan <= 0) continue;

        MargPriorBlock mb;
        mb.ambient = amb;
        mb.tangent = tan;
        mb.lin.assign(p, p + amb);      /* 线性化点必须留副本 */

        PriorBlockRef ref{PriorRefKind::CALIB, 0.0, 0};
        auto it = ref_of.find(p);
        if (it != ref_of.end()) ref = it->second;
        switch (ref.kind) {
        case PriorRefKind::CALIB:      mb.kind = MargBlockKind::CALIB;    break;
        case PriorRefKind::FRAME_POSE: mb.kind = MargBlockKind::POSE;     break;
        default:                       mb.kind = MargBlockKind::ADDITIVE; break;
        }
        spec->refs.push_back(ref);
        spec->meta.push_back(std::move(mb));
    }
    if (spec->refs.size() != spec->meta.size() || spec->meta.empty()) return false;

    int tot = 0;
    for (const MargPriorBlock &m : spec->meta) tot += m.tangent;
    if (tot != J_out.cols()) return false;   /* 列映射必须严丝合缝 */

    prior_    = spec;
    J_prior_  = J_out;
    r_prior_  = r_out;
    return true;
}

/* ============================================================================
 * GraphOptimizer::optimize
 * ==========================================================================*/
bool GraphOptimizer::optimize(std::vector<Frame> &frames, const Eigen::Vector3d &g_n) {
    if (frames.empty()) return false;

    /* ---- 1) 组装问题 ---- */
    ceres::Problem problem;
    buildProblem(problem, frames, g_n);

    /* 把边缘化先验真正加进问题。
     * 先验按块描述：pose 用 PoseManifold，calib 用 CalibManifold，
     * mix/模糊度等为加性块；块指针按身份（帧时间/模糊度键）重新解析。 */
    if (prior_ && prior_->J.rows() > 0) {
        std::vector<double*> prior_ptrs;
        if (resolvePriorPointers(frames, prior_ptrs)) {
            problem.AddResidualBlock(
                new MarginalizationPriorFactor(prior_->J, prior_->r, prior_->meta),
                nullptr, prior_ptrs);
            ++stat_prior_blocks_added;
        } else {
            /* 块对不上（例如窗口变化导致某帧已不在）→ 宁可丢弃先验也不要给错的信息 */
            if (!prior_drop_warned_) {
                std::fprintf(stderr,
                    "[Marg] 先验的参数块已无法与当前窗口对应，本次丢弃先验（信息会损失，"
                    "但比喂错信息安全）。\n");
                prior_drop_warned_ = true;
            }
            clearPrior();
        }
    }

    /* ---- 2) 求解 ---- */
    ceres::Solver::Options options;
    /* 滑窗只有 ~15 帧（≈230 维）。实测（1339 次求解，同精度同迭代数）：
     *   DENSE_SCHUR 0.51 s | SPARSE_NORMAL_CHOLESKY 0.44 s |
     *   DENSE_NORMAL_CHOLESKY 2.1 s | DENSE_QR 4.4 s
     * 所以保持原来的 DENSE_SCHUR（矩阵不大，舒尔消元比 QR 省得多）。 */
    options.linear_solver_type         = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations         = opt_.max_iterations;
    if (const char *ls = std::getenv("IPOS3_LINSOLVER")) {
        const int v = std::atoi(ls);
        if (v == 0) options.linear_solver_type = ceres::DENSE_QR;
        else if (v == 1) options.linear_solver_type = ceres::DENSE_SCHUR;
        else if (v == 2) options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
        else if (v == 3) options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    }
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    const auto t_solve_begin = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    stat_time_s += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_solve_begin).count();
    stat_solves++;
    stat_iterations += static_cast<int>(summary.iterations.size());
    if (summary.termination_type == ceres::CONVERGENCE) stat_converged++;
    last_message = summary.message;

    return summary.termination_type == ceres::CONVERGENCE;
}



size_t readSensorFile(const std::string &path,
                      std::vector<DataSensor281_t> &out,
                      size_t max_frames) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "[data_reader] 无法打开 %s\n", path.c_str());
        return 0;
    }

    std::fseek(fp, 0, SEEK_END);
    const long bytes = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);

    const size_t total = static_cast<size_t>(bytes) / sizeof(DataSensor281_t);
    const size_t n     = (max_frames == 0 || max_frames > total) ? total : max_frames;

    out.resize(n);
    const size_t read = std::fread(out.data(), sizeof(DataSensor281_t), n, fp);
    std::fclose(fp);

    std::printf("[data_reader] %s: 读入 %zu / %zu 帧 (%.2f MB)\n",
                path.c_str(), read, total,
                static_cast<double>(read * sizeof(DataSensor281_t)) / 1e6);
    return read;
}

/* 把 GNSS blh 转 NEU，不依赖 LocalFrame 的初始化顺序 ---------------- */
static Eigen::Vector3d blhToNeu(const LocalFrame &lf, const DataSensor281_t &s) {
    return lf.blh2neu(Eigen::Vector3d(s.posgps[0], s.posgps[1], s.posgps[2]));
}

/* 由双天线航向 + 水平，构造初始姿态 --------------------------------- */
static Eigen::Quaterniond initAttitudeFromYaw(double yaw) {
    /* RFU ↔ ENU，yaw=0 时对齐；这里直接给水平姿态 + 航向 */
    return a2qua(0.0, 0.0, yaw);
}

/* 双天线原始航向 -> 内部姿态 yaw --------------------------------------------
 * 与 ipos3g 的 SINSGNSSLOOSE::GNSS2Ali 完全一致：
 *   FB_B : 原样          FB_F : diffYaw(raw, 180°)
 *   LR_L : diffYaw(raw, -90°)   LR_R : diffYaw(raw, +90°)
 * 再叠加天线安装角 yaw_offset（等价于 conf_GNSSAgle）。
 * 注意 diffYaw(a,b) = a - b，所以 LR_L 是"再转 +90°"。 */
double gnssYaw2AttYaw(double raw_yaw, int ant_mode, double yaw_offset) {
    const double PI_2 = PI * 0.5;
    double r = raw_yaw;
    r = C360CC180(r);   /* 归一到 [-180,180] */
    switch (ant_mode) {
        case ANT_MODE_LR_L: r = diffYaw(r, -PI_2); break;
        case ANT_MODE_LR_R: r = diffYaw(r,  PI_2); break;
        case ANT_MODE_FB_F: r = diffYaw(r,  PI);   break;
        case ANT_MODE_FB_B:
        case ANT_MODE_ONE:
        default: break;
    }
    return diffYaw(r, -yaw_offset);
}

/* 静止检测：滑动窗口内角增量/速度增量接近 0 ------------------------ */
class StaticDetector {
public:
    explicit StaticDetector(double thr_g, double thr_a)
        : thr_g_(thr_g), thr_a_(thr_a) {}

    void push(const DataSensor281_t &s, double dt) {
        buf_.push_back(s);
        if (buf_.size() > 20) buf_.pop_front();   // 10 → 20
        (void)dt;
    }

    bool isStatic() const {
        if (buf_.size() < 10) return false;

        Eigen::Vector3d wm_mean = Eigen::Vector3d::Zero();
        Eigen::Vector3d vm_mean = Eigen::Vector3d::Zero();
        for (const auto &s : buf_) {
            wm_mean += Eigen::Vector3d(s.wm[0], s.wm[1], s.wm[2]);
            vm_mean += Eigen::Vector3d(s.vm[0], s.vm[1], s.vm[2]);
        }
        wm_mean /= buf_.size();
        vm_mean /= buf_.size();

        double wm_var = 0.0, vm_var = 0.0;
        for (const auto &s : buf_) {
            Eigen::Vector3d dw(s.wm[0], s.wm[1], s.wm[2]);
            Eigen::Vector3d dv(s.vm[0], s.vm[1], s.vm[2]);
            dw -= wm_mean; dv -= vm_mean;
            wm_var += dw.squaredNorm();
            vm_var += dv.squaredNorm();
        }
        wm_var /= buf_.size();
        vm_var /= buf_.size();

        const double wm_rms = std::sqrt(wm_var);
        const double vm_rms = std::sqrt(vm_var);

        /* 静止时加表均值模长应接近 g，这一条能剔除掉"匀速运动"被误判静止 */
        const double g_norm = vm_mean.norm();
        const double g_err  = std::fabs(g_norm - G0);   /* G0 在 app_interface.h */

        return (wm_rms < thr_g_) && (vm_rms < thr_a_) && (g_err < 0.5);
    }

private:
    std::deque<DataSensor281_t> buf_;
    double thr_g_, thr_a_;
};

RunnerStats runRealData(const std::string &bin_path,
                        const RunnerOptions &opt,
                        const std::string &out_nav_path) {
    std::vector<DataSensor281_t> raw;
    if (readSensorFile(bin_path, raw) == 0) {
        std::fprintf(stderr, "[runner] 无数据\n");
        return RunnerStats{};
    }
    return runRealData(raw, opt, out_nav_path);
}

RunnerStats runRealData(const std::vector<DataSensor281_t> &raw,
                        const RunnerOptions &opt,
                        const std::string &out_nav_path) {
    RunnerStats st;
    std::vector<Frame> trajectory;
    /* ---- 1) 数据 ---- */
    if (raw.empty()) {
        std::fprintf(stderr, "[runner] 无数据\n");
        return st;
    }
    st.n_frames = static_cast<int>(raw.size());

    /* ---- 2) 找第一个有效 GNSS，建立局部坐标系 ---- */
    int first_gnss = -1;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (isValidGnss(raw[i])) { first_gnss = static_cast<int>(i); break; }
    }
    if (first_gnss < 0) {
        std::fprintf(stderr, "[runner] 没有有效 GNSS，无法初始化\n");
        return st;
    }

    LocalFrame lf;
    lf.init(Eigen::Vector3d(raw[first_gnss].posgps[0],
                  raw[first_gnss].posgps[1],
                  raw[first_gnss].posgps[2]));


    /* ---- 3) 初始化滑窗 ---- */
    std::vector<Frame> frames;
    StaticDetector static_det(opt.static_thresh_gyr, opt.static_thresh_acc);

    GraphOptimizer::Options gopt_opt;
    gopt_opt.max_keyframes  = opt.max_keyframes;
    gopt_opt.max_iterations = 50;
    gopt_opt.whiten_preint  = opt.whiten_preint;
    gopt_opt.bias_jac       = opt.bias_jac;
    gopt_opt.fix_first_pose = opt.fix_first_pose;
    gopt_opt.odo_abv        = opt.odo_abv;
    gopt_opt.odo_lvOD       = opt.odo_lvOD;
    gopt_opt.calib_mode     = opt.calib_mode;   // ← 补这一行
    gopt_opt.use_visual    = opt.use_visual;
    gopt_opt.visual_sigma  = opt.visual_sigma;
    gopt_opt.visual_huber  = opt.visual_huber;
    gopt_opt.inv_depth_min = 1.0 / 500.0;
    gopt_opt.inv_depth_max = 1.0 / 0.5;

    gopt_opt.use_gnss_dd        = opt.use_gnss_dd;
    gopt_opt.fix_ambiguity      = opt.fix_ambiguity;
    gopt_opt.max_freq           = opt.max_freq;
    gopt_opt.use_wl_nl          = opt.use_wl_nl;
    gopt_opt.use_if_combination = opt.use_if_combination;
    gopt_opt.wl_sigma           = opt.wl_sigma;
    gopt_opt.nl_sigma           = opt.nl_sigma;
    gopt_opt.if_sigma           = opt.if_sigma;
    gopt_opt.gnss_ant_lever     = opt.gnss_ant_lever;

    if (const char *it = std::getenv("IPOS3_MAX_ITER")) {
        gopt_opt.max_iterations = std::atoi(it);
    }
    GraphOptimizer gopt(gopt_opt);

    std::map<double, std::vector<VisualObs>> visual_table;
    if (opt.use_visual && !opt.cam_dir.empty()) {
        visual_table = preprocessImages(opt.cam_dir, opt.cam_model,
                                        opt.visual_sigma);
    }

    /* 标定参数初值（标定与导航通用） */
    CalibState calib = opt.calib_init;
    /* 兼容旧配置：如果 calib_init 未被填过，从 odo_* 与 yaw_offset 派生 */
    if (calib.sodo == 0.0 && calib.abv_pitch == 0.0 && calib.abv_yaw == 0.0 &&
        calib.lvOD.x() == 0.0 && calib.lvOD.y() == 0.0 && calib.lvOD.z() == 0.0 &&
        calib.yaw_gnss_offset == 0.0) {
        calib.sodo           = opt.odo_sodo_init;
        calib.abv_pitch      = opt.odo_abv.x();
        calib.abv_yaw        = opt.odo_abv.y();
        calib.lvOD           = opt.odo_lvOD;
        calib.yaw_gnss_offset = opt.yaw_offset;
    }
    gopt.setCalibState(calib);

    gopt.setFrameContext(lf.R_en_m3(), lf.p0Ecef_v3());
    gopt.setBasePosEcef(opt.base_pos_ecef);

    auto evalFrameCompare = [&gopt](const Frame &f) {
        FrameCompare c;
        c.time = f.time;

        /* --- 位置 --- */
        if (f.has_gnss) {
            c.has_gnss_pos = true;
            const double pe = f.pose[0] - f.gnss_pos.x();
            const double pn = f.pose[1] - f.gnss_pos.y();
            const double pu = f.pose[2] - f.gnss_pos.z();
            c.dpe = pe; c.dpn = pn; c.dpu = pu;
            c.dp_xy = std::sqrt(pe * pe + pn * pn);
            c.dp_3d = std::sqrt(pe * pe + pn * pn + pu * pu);
        }

        /* --- 速度 --- */
        if (f.has_gnss_vel) {
            c.has_gnss_vel = true;
            const double ve = f.mix[0] - f.gnss_vn.x();
            const double vn = f.mix[1] - f.gnss_vn.y();
            const double vu = f.mix[2] - f.gnss_vn.z();
            c.dve = ve; c.dvn = vn; c.dvu = vu;
            c.dv_xy = std::sqrt(ve * ve + vn * vn);
            c.dv_3d = std::sqrt(ve * ve + vn * vn + vu * vu);
        }

        /* --- 航向：raw_yaw_conv 只含 ant_mode 折算，需叠加 yaw_off_calib --- */
        if (f.has_raw_yaw) {
            const double yoff = gopt.getCalibState().yaw_gnss_offset;  // ← 实时
            c.has_gnss_yaw = true;
            Eigen::Quaterniond q(f.pose[6], f.pose[3], f.pose[4], f.pose[5]);
            double yaw_est = q2att(q).z();
            double yaw_ref = diffYaw(f.raw_yaw_conv + yoff, 0.0);
            c.dyaw_rad = diffYaw(yaw_est, yaw_ref);
            c.dyaw_deg = c.dyaw_rad / DEG;
        }

        return c;
    };

    ErrorAccumulator acc_init;   /* 初值 vs GNSS */
    ErrorAccumulator acc_opt;    /* 优化后 vs GNSS */

    /* 记录某帧的对比结果（初值/优化后共用一份 FrameCompare 表），
     * 并顺便累积"估计航向 vs 双天线航向"的统计量。 */
    double yaw_sum = 0.0, yaw_sq = 0.0;
    int    yaw_n   = 0;
    auto recordFrame = [&](const Frame &f) {
        const FrameCompare c = evalFrameCompare(f);
        auto it = st.per_kf_time_to_idx.find(f.time);
        if (it != st.per_kf_time_to_idx.end()) {
            st.per_kf[it->second] = c;
        } else {
            st.per_kf_time_to_idx[f.time] = st.per_kf.size();
            st.per_kf.push_back(c);
        }
        if (c.has_gnss_yaw) {
            yaw_sum += c.dyaw_deg;
            yaw_sq  += c.dyaw_deg * c.dyaw_deg;
            yaw_n++;
        }
    };

    /* 汇总用的轻量版（只关心位置模长，用于统计）*/
    auto evalOneFrame = [](const Frame &f, ErrorAccumulator &acc) {
        if (!f.has_gnss) return;
        Eigen::Vector3d p(f.pose[0], f.pose[1], f.pose[2]);
        double e = norm(p - f.gnss_pos);
        acc.add(f.time, e);
    };

    std::printf("[runner] %s模式 | 标定初值: sodo=%.4f, abv=(%.2f, %.2f)°, lvOD=(%.3f, %.3f, %.3f), yaw_off=%.3f°\n",
                opt.calib_mode ? "标定" : "导航",
                calib.sodo,
                calib.abv_pitch / DEG, calib.abv_yaw / DEG,
                calib.lvOD.x(), calib.lvOD.y(), calib.lvOD.z(),
                calib.yaw_gnss_offset / DEG);

    gopt.clearPrior();

    /* 第一帧：姿态取第一个有效 yaw，位置 NEU=0 */
    {
        Frame f0;
        f0.time = raw[first_gnss].t;

        /* 初始姿态：有双天线航向就用它（按天线安装方式折算），
         * 否则只能给 yaw=0（水平姿态）。这里以前被 use_gnss_yaw 一起关掉，
         * 导致默认配置下初始航向恒为 0，和真实航向差 90° 以上。 */
        double yaw0 = 0.0;
        /* 注意单位：yaw 是 rad（0~2π 回绕），yawrms 是 deg（原工程 YAW_ALIGN=5.0 deg，
         * 179.99 之类的值表示航向无效）。以前把 yawrms 当 rad 判 5*DEG，会全部拒掉。 */
        const bool yaw0_ok = (raw[first_gnss].yawrms > 0.0 &&
                              raw[first_gnss].yawrms < 5.0 &&
                              std::fabs(raw[first_gnss].yaw) > 1e-6);
        bool used_gnss_yaw0 = false;
        if (opt.init_yaw_from_gnss && yaw0_ok && opt.ant_mode != ANT_MODE_ONE) {
            yaw0 = gnssYaw2AttYaw(raw[first_gnss].yaw, opt.ant_mode, opt.yaw_offset);
            used_gnss_yaw0 = true;
        }
        Eigen::Quaterniond q0 = initAttitudeFromYaw(yaw0);
        std::printf("[runner] 初始航向: %.3f deg (%-5s, 双天线原始 %.3f deg)\n",
                    yaw0 / DEG, used_gnss_yaw0 ? "GNSS" : "默认0",
                    raw[first_gnss].yaw / DEG);

        f0.pose = { 0, 0, 0, q0.x(), q0.y(), q0.z(), q0.w() };
        f0.mix = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };   /* mix 为 {v,bg,ba}；sodo 在 calib[0] */
        f0.has_gnss = true;
        f0.gnss_pos = Eigen::Vector3d(0, 0, 0);   /* 局部系原点 */
        f0.gnss_std = Eigen::Vector3d(opt.gnss_pos_scale * std::max(raw[first_gnss].posstd[0], 0.5),
                            opt.gnss_pos_scale * std::max(raw[first_gnss].posstd[1], 0.5),
                            opt.gnss_pos_scale * std::max(raw[first_gnss].posstd[2], 1.0));
        frames.push_back(f0);
        st.n_gnss_used++;
    }

    PreintegrationParam pp;
    pp.gyr_arw      = 0.5 * DEG / 60.0;    /* 0.5 deg/sqrt(hr) */
    pp.acc_vrw      = 0.1 * MG;            /* 0.1 mg/sqrt(Hz) = 0.1e-3·g m/s^1.5 */
    pp.gyr_bias_std = 10.0 * DPH;          /* 10 deg/hr */
    pp.acc_bias_std = 1.0 * MG;            /* 1 mg */
    pp.corr_time    = 3600.0;
    pp.abv  = opt.odo_abv;
    pp.lvOD = opt.odo_lvOD;

    Preintegration preint(pp);
    preint.setBias(O31, O31);

    double t_last_kf = raw[first_gnss].t;
    double t_last_imu = t_last_kf;
    double t_preint = 0.0;               /*< 预积分累计耗时 */
    Eigen::Vector3d  bg_cur(O31), ba_cur(O31);     /*< 下一段预积分用的零偏 */

    /* GNSS 中断区间（相对首帧），用于考核航位推算 */
    const double t0_run = raw[first_gnss].t;
    const bool   use_gap = (opt.gnss_gap_t1 > opt.gnss_gap_t0);
    auto inGap = [&](double tt) {
        return use_gap && (tt - t0_run) >= opt.gnss_gap_t0 && (tt - t0_run) <= opt.gnss_gap_t1;
    };
    ErrorAccumulator acc_gap;

    
    int n_by_time = 0, n_by_gnss = 0, n_both = 0;
    double t_prev_kf = 0.0;
    int n_short = 0, n_long = 0;
    /* ---- 4) 主循环 ---- */
        /* ---- 4) 主循环 ---- */
    for (size_t i = first_gnss + 1; i < raw.size(); ++i) {
        const auto &s = raw[i];
        const double dt = s.t - t_last_imu;
        if (dt <= 1e-9 || dt > 0.5) {
            t_last_imu = s.t;
            continue;
        }

        static_det.push(s, dt);

        ImuMeas m;
        m.time   = s.t;
        m.dt     = dt;
        m.dtheta = Eigen::Vector3d(s.wm[0], s.wm[1], s.wm[2]);
        m.dvel   = Eigen::Vector3d(s.vm[0], s.vm[1], s.vm[2]);
        if (opt.use_odometer && s.dS < 90.0) {
            m.odovel = s.dS;   // ← 补这一行
        }

        const auto t_int_begin = std::chrono::steady_clock::now();
        preint.integration(m);

        t_preint += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_int_begin).count();
        t_last_imu = s.t;

        const bool kf_by_time  = (s.t - t_last_kf >= opt.kf_dt);
        const bool gnss_valid   = isValidGnss(s);
        const bool gnss_in_gap  = gnss_valid && inGap(s.t);
        const bool has_gnss_now = gnss_valid && !gnss_in_gap;
        if (!kf_by_time && !has_gnss_now) continue;

        /* ---- 诊断 ---- */
        if (kf_by_time && has_gnss_now)  n_both++;
        else if (kf_by_time)              n_by_time++;
        else                              n_by_gnss++;

        double dt_kf = s.t - t_prev_kf;
        if (dt_kf < 1.5) n_short++;
        else             n_long++;
        t_prev_kf = s.t;

        if (st.n_kf_total % 50 == 0) {
            printf("[KF#%4d] t=%.3f dt_kf=%.3f reason=%s%s\n",
                st.n_kf_total, s.t, dt_kf,
                kf_by_time ? "TIME" : "",
                has_gnss_now ? " GNSS" : "");
        }

        /* 4.4 构造新关键帧：先取上一帧快照（值拷贝，避免悬垂） */
        const Frame fprev = frames.back();

        const Eigen::Vector3d dv_b0 = preint.v();
        const Eigen::Quaterniond   dq_b0 = preint.q();
        const Eigen::Vector3d  dp_b0 = preint.p();
        const double T_k   = preint.dt();

        Eigen::Quaterniond q_prev(fprev.pose[6], fprev.pose[3], fprev.pose[4], fprev.pose[5]);
        Eigen::Vector3d v_prev(fprev.mix[0], fprev.mix[1], fprev.mix[2]);
        Eigen::Vector3d p_prev(fprev.pose[0], fprev.pose[1], fprev.pose[2]);

        Eigen::Quaterniond q_pred = q_prev * dq_b0;
        normlize(q_pred);
        Eigen::Matrix3d R_prev = q_prev.toRotationMatrix();
        Eigen::Vector3d g_n = lf.earthModel().gn;
        Eigen::Vector3d v_pred = v_prev
                               + R_prev * dv_b0
                               + g_n * T_k;
        Eigen::Vector3d p_pred = p_prev
                               + v_prev * T_k
                               + 0.5 * g_n * T_k * T_k
                               + R_prev * dp_b0;

        Frame fk;
        fk.time = s.t;
        fk.pose = { p_pred(0), p_pred(1), p_pred(2),
                    q_pred.x(), q_pred.y(), q_pred.z(), q_pred.w() };
        fk.mix  = { v_pred(0), v_pred(1), v_pred(2),
                    fprev.mix[3], fprev.mix[4], fprev.mix[5],
                    fprev.mix[6], fprev.mix[7], fprev.mix[8] };

        /* 4.5 观测填充 */
        if (has_gnss_now) {
            fk.has_gnss = true;
            fk.gnss_pos = blhToNeu(lf, s);
            fk.gnss_std = Eigen::Vector3d(opt.gnss_pos_scale * std::max(s.posstd[0], 0.5),
                                opt.gnss_pos_scale * std::max(s.posstd[1], 0.5),
                                opt.gnss_pos_scale * std::max(s.posstd[2], 1.0));
            st.n_gnss_used++;
        } else if (gnss_in_gap) {
            fk.gnss_pos    = blhToNeu(lf, s);
            fk.gnss_in_gap = true;
        }

        /* --- GNSS 速度因子 --- */
        if (opt.use_gnss_vel && !gnss_in_gap) {
            const double vn_norm = std::sqrt(s.vngps[0]*s.vngps[0] +
                                             s.vngps[1]*s.vngps[1] +
                                             s.vngps[2]*s.vngps[2]);
            if (vn_norm > 0.1) {
                const double sat_scale = (s.satnum > 1.0)
                    ? std::max(1.0, 8.0 / s.satnum) : 1.0;
                fk.has_gnss_vel = true;
                fk.gnss_vn     = Eigen::Vector3d(s.vngps[0], s.vngps[1], s.vngps[2]);
                fk.gnss_vn_std = Eigen::Vector3d(opt.gnss_vel_scale * 0.1 * sat_scale,
                                       opt.gnss_vel_scale * 0.1 * sat_scale,
                                       opt.gnss_vel_scale * 0.2 * sat_scale);
            }
        }

        /* --- 双天线航向 --- */
        if (opt.ant_mode != ANT_MODE_ONE && !gnss_in_gap &&
            s.yawrms > 0.0 && s.yawrms < 5.0 &&
            std::fabs(s.yaw) > 1e-6) {
            /* 只做 ant_mode 折算（±90° / 180°），yaw_offset 交给因子内部 */
            const double yaw_ant = gnssYaw2AttYaw(s.yaw, opt.ant_mode, 0.0);

            fk.has_raw_yaw  = true;
            fk.raw_yaw_conv = yaw_ant;   /* 用于统计时需叠加上 calib 里的 yaw_off */
            if (opt.use_gnss_yaw) {
                fk.has_gnss_yaw = true;
                fk.gnss_yaw     = yaw_ant;
                fk.gnss_yaw_std = std::max(s.yawrms * DEG, 0.5 * DEG);
            }
        }

        /* --- 里程计 --- */
        if (opt.use_odometer && !gnss_in_gap &&
            s.dS > 1e-6 && s.dS < 90.0) {
            fk.has_odo        = true;
            fk.odo_dS         = s.dS;
            fk.odo_dt         = dt;
            fk.odo_omega_meas = Eigen::Vector3d(s.wm[0] / dt, s.wm[1] / dt, s.wm[2] / dt);
            fk.odo_std        = opt.odo_vel_scale * opt.odo_vel_std;
            st.n_odo_used++;
        }

        /* ---- 视觉观测：按最近邻时间戳匹配 ---- */
        if (opt.use_visual && !visual_table.empty()) {
            auto it = visual_table.lower_bound(s.t);
            double best_dt = 1e9;
            const std::vector<VisualObs>* best = nullptr;
            if (it != visual_table.end()) {
                best_dt = it->first - s.t;
                best    = &it->second;
            }
            if (it != visual_table.begin()) {
                auto it2 = std::prev(it);
                if (s.t - it2->first < best_dt) {
                    best_dt = s.t - it2->first;
                    best    = &it2->second;
                }
            }
            if (best && best_dt < 0.05) {   /* 50 ms 内有效 */
                fk.visual_obs = *best;
            }
        }

        /* 4.6 挂预积分：用 frames.back() 而不是 fprev */
        /*    冻结时丢掉重放历史（每段 ~200 条 ImuMeas），拷贝代价随之消失 */
        Preintegration preint_frozen = preint;
        preint_frozen.dropHistory();
        frames.back().pre_to_next = std::make_shared<Preintegration>(preint_frozen);

        /* 4.7 记录初值误差（push 之前，fk 即预积分外推得到的初值） */
        evalOneFrame(fk, acc_init);
        /* 4.7 推入新帧 */
        frames.push_back(fk);
        st.n_kf_total++;

        /* 关键帧生成时 */
        size_t idx = st.per_kf.size();
        st.per_kf.push_back(evalFrameCompare(fk));           /* 初值版本 */
        st.per_kf_time_to_idx[fk.time] = idx;

        /* 滑出时（用 time 找到下标，覆盖） */
        auto it = st.per_kf_time_to_idx.find(frames.front().time);
        if (it != st.per_kf_time_to_idx.end()) {
            st.per_kf[it->second] = evalFrameCompare(frames.front());  /* 优化后版本 */
        }

        if (st.n_kf_total % 100 == 0) {   /* 每 100 帧打印一次 */
            printf("kf#%4d  t=%.1f  p=(%8.3f, %8.3f, %8.3f)  "
                "v=(%6.3f, %6.3f, %6.3f)  yaw=%.2f°  %s\n",
                st.n_kf_total, fk.time,
                fk.pose[0], fk.pose[1], fk.pose[2],
                fk.mix[0],  fk.mix[1],  fk.mix[2],
                q2att(Eigen::Quaterniond(fk.pose[6], fk.pose[3], fk.pose[4], fk.pose[5])).z() / DEG,
                fk.is_static ? "STATIC" : "moving");
        }
        
        t_last_kf = s.t;

        /* 4.8 重置预积分器（零偏在 4.11 用最新估计回灌） */
        preint.reset();

        /* 4.9 滑窗求解
         *   因子约定：g_n 为真实重力矢量（ENU 下指向下，即 gn 本身）。
         *   原来只在窗口攒满 15 帧时才求解，于是前 14 s 完全是惯导外推，
         *   20 mg 的加计零偏会让这段初值误差累积到 ~20 m。改为只要窗口里
         *   有 2 帧就解一次，稳态下求解次数不变（窗口满后每帧本来就解一次）。 */
        const int min_solve = std::max(2, gopt.options().min_frames_solve);
        const bool solved = (static_cast<int>(frames.size()) >= min_solve);
        if (solved) {
            gopt.optimize(frames, lf.earthModel().gn);
        }

        /* 4.10 滑窗超限：最旧帧出窗 */
        if (static_cast<int>(frames.size()) >= gopt.options().max_keyframes) {
            recordFrame(frames.front());

            /* 优化后、erase 之前，记录被挤出窗口的最旧帧 */
            evalOneFrame(frames.front(), acc_opt);   /* ← 新增 */
            if (frames.front().gnss_in_gap) {
                const Frame &fg = frames.front();
                acc_gap.add(fg.time,
                            norm(Eigen::Vector3d(fg.pose[0], fg.pose[1], fg.pose[2]) - fg.gnss_pos));
            }

            /* 优化后、erase 之前，保存最旧帧到完整轨迹 */
            trajectory.push_back(frames.front());

            /* 出窗前先做边缘化：把 frames[0] 的信息通过 Schur 补保留成先验，
             * 而不是直接丢掉（原先这里只有 erase，属于"信息丢失型"滑窗）。 */
            if (!gopt.marginalizeOldestFrame(frames, lf.earthModel().gn)) {
                static bool marg_drop_warned = false;
                if (!marg_drop_warned) {
                    std::fprintf(stderr, "[Marg] 边缘化未成功，本步按丢帧处理（信息会损失）。\n");
                    marg_drop_warned = true;
                }
            }

            frames.erase(frames.begin());
        }

        /* 4.11 把最新零偏估计回灌给下一段预积分
         *      预积分在积分时就把零偏扣掉，外推出的初值才是零偏补偿后的结果；
         *      它同时是因子中零偏一阶展开的线性化点。 */
        if (solved && !frames.empty()) {
            const Frame &fnew = frames.back();
            bg_cur = Eigen::Vector3d(fnew.mix[3], fnew.mix[4], fnew.mix[5]);
            ba_cur = Eigen::Vector3d(fnew.mix[6], fnew.mix[7], fnew.mix[8]);
            preint.setBias(bg_cur, ba_cur);
        }
    }

    /* ---- 5) 收尾：对最后一批关键帧跑一次优化 ---- */
    if (!frames.empty()) {
        const Eigen::Vector3d g_vect = lf.earthModel().gn;
        gopt.optimize(frames, g_vect);

        /* 末尾残留帧也加入轨迹 */
        for (const auto &f : frames) {
            evalOneFrame(f, acc_opt);                /* ← 新增 */
            if (f.gnss_in_gap) {
                acc_gap.add(f.time, norm(Eigen::Vector3d(f.pose[0], f.pose[1], f.pose[2]) - f.gnss_pos));
            }
            recordFrame(f);
            trajectory.push_back(f);            // ← 新增
        }
    }

    /* 记录最终标定状态 */
    st.calib_final = gopt.getCalibState();
    std::printf("\n========== 标定参数（%s模式）==========\n",
                opt.calib_mode ? "标定" : "导航");
    std::printf("  刻度因子 sodo   : %+10.6f\n", st.calib_final.sodo);
    std::printf("  安装角 pitch    : %+10.4f deg\n", st.calib_final.abv_pitch / DEG);
    std::printf("  安装角 yaw      : %+10.4f deg\n", st.calib_final.abv_yaw   / DEG);
    std::printf("  杆臂   lvOD     : (%+8.4f, %+8.4f, %+8.4f) m\n",
                st.calib_final.lvOD.x(), st.calib_final.lvOD.y(), st.calib_final.lvOD.z());
    std::printf("  航向偏置        : %+10.4f deg\n", st.calib_final.yaw_gnss_offset / DEG);

    st.n_kf          = static_cast<int>(frames.size());
    st.n_prior_added = gopt.stat_prior_blocks_added;
    st.total_time = raw.back().t - raw.front().t;
    st.final_cost = gopt.hasPrior() ? gopt.r_prior().norm() : 0.0;

    /* ---- 6) GNSS 对比统计 ---- */
    st.n_cmp_init = acc_init.size();
    st.n_cmp_opt  = acc_opt.size();
    acc_init.fill(st.init_max, st.init_mean, st.init_rms, st.init_std, st.init_pct);
    acc_opt .fill(st.opt_max,  st.opt_mean,  st.opt_rms,  st.opt_std,  st.opt_pct);
    if (st.opt_rms > 0.0) {
        st.improve_ratio = st.init_rms / st.opt_rms;
    }

    /* ---- 6.1) GNSS 中断期间的航位推算误差 ---- */
    st.n_gap_frames = acc_gap.size();
    for (int i = 0; i < acc_gap.size(); ++i) {
        st.gap_max_err = std::max(st.gap_max_err, acc_gap.err[i]);
        st.gap_end_err = acc_gap.err[i];
    }

    /* ---- 6.2) 计算量统计 ---- */
    st.t_preint     = t_preint;
    st.t_solve      = gopt.stat_time_s;
    st.n_solve      = gopt.stat_solves;
    st.n_solve_ok   = gopt.stat_converged;
    st.n_solve_iter = gopt.stat_iterations;

    /* ---- 6.3) 估计航向 vs 双天线航向 ---- */
    if (yaw_n > 0) {
        st.n_yaw_cmp     = yaw_n;
        st.yaw_diff_mean = yaw_sum / yaw_n;
        st.yaw_diff_rms  = std::sqrt(yaw_sq / yaw_n);
    }

    /* ---- 7) 打印 ---- */
    std::printf("\n[runner] 累计关键帧 = %d, 窗口残留 = %d\n",
                st.n_kf_total, st.n_kf);
    std::printf("[runner] GNSS 使用 = %d, 静止帧 = %d, 时长 = %.1f s\n",
                st.n_gnss_used, st.n_static_frames, st.total_time);

    std::printf("\n========== GNSS 对比（位置误差，单位 m） ==========\n");
    std::printf("  ---- 初值（预积分递推，未优化）----\n");
    std::printf("     样本 = %d\n", st.n_cmp_init);
    std::printf("     max = %8.3f   mean = %8.3f   rms = %8.3f   std = %8.3f\n",
                st.init_max, st.init_mean, st.init_rms, st.init_std);
    std::printf("     P50 = %8.3f   P90 = %8.3f   P95 = %8.3f   P99 = %8.3f\n",
                st.init_pct[0], st.init_pct[1], st.init_pct[2], st.init_pct[3]);

    std::printf("  ---- 优化后 ----\n");
    std::printf("     样本 = %d\n", st.n_cmp_opt);
    std::printf("     max = %8.3f   mean = %8.3f   rms = %8.3f   std = %8.3f\n",
                st.opt_max, st.opt_mean, st.opt_rms, st.opt_std);
    std::printf("     P50 = %8.3f   P90 = %8.3f   P95 = %8.3f   P99 = %8.3f\n",
                st.opt_pct[0], st.opt_pct[1], st.opt_pct[2], st.opt_pct[3]);

    std::printf("  ---- 改善比 = %.2f× ----\n", st.improve_ratio);

    if (!out_nav_path.empty()) {
        FILE *fp = std::fopen(out_nav_path.c_str(), "w");
        if (fp) {
            std::fprintf(fp, "# t, pE, pN, pU, vE, vN, vU, qx, qy, qz, qw\n");
            for (const auto &f : trajectory) {
                std::fprintf(fp,
                    "%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, "
                    "%.6f, %.6f, %.6f, %.6f\n",
                    f.time,
                    f.pose[0], f.pose[1], f.pose[2],
                    f.mix[0],  f.mix[1],  f.mix[2],
                    f.pose[3], f.pose[4], f.pose[5], f.pose[6]);
            }
            std::fclose(fp);
            std::printf("[runner] 已写出 %s\n", out_nav_path.c_str());
        }
    }

        /* ---- 输出逐帧对比 CSV ---- */
    if (!opt.compare_csv_path.empty()) {
        FILE *fp = std::fopen(opt.compare_csv_path.c_str(), "w");
        if (fp) {
            std::fprintf(fp,
                "# t, dpe, dpn, dpu, dp_xy, dp_3d, "
                "dve, dvn, dvu, dv_xy, dv_3d, dyaw_deg\n");
            for (const auto &c : st.per_kf) {
                std::fprintf(fp,
                    "%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    c.time,
                    c.dpe, c.dpn, c.dpu, c.dp_xy, c.dp_3d,
                    c.dve, c.dvn, c.dvu, c.dv_xy, c.dv_3d,
                    c.dyaw_deg);
            }
            std::fclose(fp);
            std::printf("[runner] 逐帧对比已写出: %s (%zu 行)\n",
                        opt.compare_csv_path.c_str(), st.per_kf.size());
        } else {
            std::fprintf(stderr, "[runner] 无法写出 %s\n",
                         opt.compare_csv_path.c_str());
        }
    }
    std::printf("\n========== 关键帧触发统计 ==========\n");
    std::printf("  by TIME only : %d\n", n_by_time);
    std::printf("  by GNSS only : %d\n", n_by_gnss);
    std::printf("  by both      : %d\n", n_both);
    std::printf("  dt_kf < 1.5s : %d\n", n_short);
    std::printf("  dt_kf >=1.5s : %d\n", n_long);

    std::printf("\n========== 频率校验 ==========\n");
    const double dur = raw.back().t - raw.front().t;
    std::printf("  时长        : %.1f s\n", dur);
    std::printf("  累计关键帧  : %d  (%.2f Hz)\n",
                st.n_kf_total,
                st.n_kf_total / dur);
    std::printf("  per_kf 条数 : %zu (%.2f Hz)\n",
                st.per_kf.size(),
                st.per_kf.size() / dur);
    std::printf("  GNSS 使用   : %d  (%.2f Hz)\n",
                st.n_gnss_used,
                st.n_gnss_used / dur);

    if (st.n_gap_frames > 0) {
        std::printf("\n========== GNSS 中断（纯航位推算）==========\n");
        std::printf("  中断区间    : [%.1f, %.1f] s\n", opt.gnss_gap_t0, opt.gnss_gap_t1);
        std::printf("  样本        : %d 帧\n", st.n_gap_frames);
        std::printf("  最大误差    : %.3f m\n", st.gap_max_err);
        std::printf("  结束时刻误差: %.3f m\n", st.gap_end_err);
    }

    if (st.n_yaw_cmp > 0) {
        std::printf("\n========== 航向一致性（估计 - 双天线，折算后）==========\n");
        std::printf("  样本        : %d 帧\n", st.n_yaw_cmp);
        std::printf("  平均偏差    : %8.2f deg   (系统性偏差，接近 0 说明 ant_mode 选对了)\n",
                    st.yaw_diff_mean);
        std::printf("  均方根偏差  : %8.2f deg\n", st.yaw_diff_rms);
    }

    std::printf("\n========== 计算量 ==========\n");
    std::printf("  预积分      : %7.3f s  (%5.1f%%)\n", st.t_preint,
                (st.t_preint + st.t_solve) > 0 ? 100.0 * st.t_preint /
                (st.t_preint + st.t_solve) : 0.0);
    std::printf("  图优化求解  : %7.3f s  (%5.1f%%)   [%d 次, 收敛 %d, 共 %d 次迭代, 平均 %.1f 次/解]\n",
                st.t_solve,
                (st.t_preint + st.t_solve) > 0 ? 100.0 * st.t_solve /
                (st.t_preint + st.t_solve) : 0.0,
                st.n_solve, st.n_solve_ok, st.n_solve_iter,
                st.n_solve > 0 ? static_cast<double>(st.n_solve_iter) / st.n_solve : 0.0);

    return st;
}

/* ============================================================================
 * 以下为原 src/amb_ils.h 的内容（模糊度整数最小二乘），已并入本头文件
 * ==========================================================================*/
/* ============================================================================
 * amb_ils.cpp —— 模糊度整数最小二乘（实现说明见本文件顶部 amb_ils 段）
 *
 * 搜索用的是标准（Teunissen）形式：
 *     Qz⁻¹ = RᵀR  （R 上三角，Cholesky）
 *     χ²(z) = ‖R (z − ẑ)‖² = Σ_i s_i²,   s_i = R(i,i)(z_i − ẑ_i) + Σ_{j>i} R(i,j)(z_j − ẑ_j)
 * 从 i = n−1 **递减**扫描：第 i 步的 Σ_{j>i} 只依赖已经定下来的 z_j，
 * 所以每层就是一个一维区间：
 *     中心 c_i = ẑ_i − t_i / R(i,i) ,  t_i = Σ_{j>i} R(i,j)(z_j − ẑ_j)
 *     半径 r_i = sqrt(rem) / R(i,i)
 * 这个形式比用 LDLᵀ 的条件均值更不容易写错（早期版本把半径写成 sqrt(rem/D_i)、
 * 中心写成 Σ L(i,j)d_j，n≥3 时是错的，导致报出的 χ² 根本不是真实二次型）。
 * ==========================================================================*/



/* LDLᵀ：Q = L D Lᵀ，L 为单位下三角。返回 false 表示 D 出现非正数。 */
static bool ldltUnit(const Eigen::MatrixXd &Q, Eigen::MatrixXd &L, Eigen::VectorXd &D) {
    const int n = static_cast<int>(Q.rows());
    L = Eigen::MatrixXd::Identity(n, n);
    D.resize(n);
    for (int i = 0; i < n; ++i) {
        double d = Q(i, i);
        for (int k = 0; k < i; ++k) d -= L(i, k) * L(i, k) * D(k);
        if (!(d > 0.0)) return false;
        D(i) = d;
        for (int j = i + 1; j < n; ++j) {
            double s = Q(j, i);
            for (int k = 0; k < i; ++k) s -= L(j, k) * L(i, k) * D(k);
            L(j, i) = s / d;
        }
    }
    return true;
}

/* Q⁻¹ = RᵀR 的上三角 R（失败表示非正定） */
static bool cholInvUpper(const Eigen::MatrixXd &Q, Eigen::MatrixXd &R) {
    const int n = static_cast<int>(Q.rows());
    Eigen::LLT<Eigen::MatrixXd> llt(Q);
    if (llt.info() != Eigen::Success) return false;
    const Eigen::MatrixXd Winv = llt.solve(Eigen::MatrixXd::Identity(n, n));
    Eigen::LLT<Eigen::MatrixXd> llt2(Winv);
    if (llt2.info() != Eigen::Success) return false;
    R = llt2.matrixU();                       /* Winv = Rᵀ R */
    return (R.diagonal().array() > 0.0).all();
}

static double normCdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

/* χ²(z) = ‖R(z − ẑ)‖² */
static double quadFormR(const Eigen::MatrixXd &R, const Eigen::VectorXd &z,
                        const Eigen::VectorXd &a) {
    const Eigen::VectorXd d = z - a;
    return (R * d).squaredNorm();
}

/* 去相关：整数 Gauss 变换（把 |L(k,i)| 压到 0.5 以内）+ 按条件方差升序重排。
 * 只维护整数变换 T，Qt = Tᵀ Q T、at = Tᵀ a 每次都从 T 重算，避免增量更新出错。
 * 关键性质：所用的 Gauss 变换保持 D 不变。 */
static int decorrelate(const Eigen::MatrixXd &Q, Eigen::MatrixXd &T,
                       Eigen::MatrixXd &Qt, Eigen::VectorXd &at,
                       const Eigen::VectorXd &a, int max_pass = 30) {
    const int n = static_cast<int>(Q.rows());
    T  = Eigen::MatrixXd::Identity(n, n);   /* z = Tᵀ a 的 T */
    Qt = Q;
    at = a;
    int n_gauss = 0;

    for (int pass = 0; pass < max_pass; ++pass) {
        Eigen::MatrixXd L;
        Eigen::VectorXd D;
        if (!ldltUnit(Qt, L, D)) break;
        bool changed = false;

        for (int k = n - 1; k >= 1; --k) {
            for (int i = k - 1; i >= 0; --i) {
                if (!ldltUnit(Qt, L, D)) break;
                const double mu = std::round(L(k, i));
                if (mu == 0.0) continue;
                /* 取 G = I − μ·e_i e_kᵀ（G(i,k) = −μ），Qz = Gᵀ Q G 满足
                 *   L(k,i) ← L(k,i) − μ 且 D 不变
                 * 这正是 LAMBDA 去相关的关键性质；方向写反会改动 D，
                 * 反而把条件数搞差、枚举爆掉。T ← T·G ⇒ T.col(k) −= μ·T.col(i)。 */
                T.col(k) -= mu * T.col(i);
                Qt = T.transpose() * Q * T;
                at = T.transpose() * a;
                ++n_gauss;
                changed = true;
            }
        }

        /* 按条件方差升序重排（置换矩阵同样是幺模整数变换） */
        if (!ldltUnit(Qt, L, D)) break;
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int p, int q) { return D(p) < D(q); });
        bool sorted = true;
        for (int i = 0; i < n; ++i) if (idx[i] != i) { sorted = false; break; }
        if (!sorted) {
            Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n, n);
            for (int i = 0; i < n; ++i) P(i, idx[i]) = 1.0;
            /* 目标：新下标 j 对应旧下标 idx[j]，即 at_new = P·at_old（P(j,idx[j])=1）。
             * 由 Qt = Tᵀ Q T、at = Tᵀ a 可知必须 T ← T·Pᵀ。
             * Ti 不再增量维护：最后一次性求 T⁻ᵀ 并做整数校验（见 ilsResolve）。 */
            T = T * P.transpose();
            Qt = T.transpose() * Q * T;
            at = T.transpose() * a;
            changed = true;
        }

        if (!changed) break;
    }
    return n_gauss;
}

bool ilsResolve(const Eigen::VectorXd &a, const Eigen::MatrixXd &Q,
                IlsResult *out, double ratio_thresh, long max_nodes) {
    const int n = static_cast<int>(a.size());
    IlsResult r;
    r.n = n;
    r.a_float = a;
    if (n <= 0 || Q.rows() != n || Q.cols() != n) { if (out) *out = r; return false; }

    r.cond_before = Q.diagonal().maxCoeff() / std::max(1e-300, Q.diagonal().minCoeff());

    /* 条件标准差（原空间序）：原始 Q 的 LDLᵀ 的 D 开方 */
    {
        Eigen::MatrixXd L0;
        Eigen::VectorXd D0;
        if (ldltUnit(Q, L0, D0)) {
            r.cond_sigma.resize(n);
            for (int i = 0; i < n; ++i) r.cond_sigma(i) = std::sqrt(std::max(0.0, D0(i)));
        }
    }

    Eigen::MatrixXd T, Qt;
    Eigen::VectorXd at;
    r.n_decorr = decorrelate(Q, T, Qt, at, a);
    r.cond_after = Qt.diagonal().maxCoeff() / std::max(1e-300, Qt.diagonal().minCoeff());

    Eigen::MatrixXd R;
    if (!cholInvUpper(Qt, R)) { if (out) *out = r; return false; }

    /* 一次性求 Ti = T⁻ᵀ，并做**整数校验**：Ti·Tᵀ == I 且 |det T| == 1。
     * 校验不过就判不可信（宁可不固定，也不给可能错的解）。 */
    Eigen::MatrixXd Ti;
    {
        const Eigen::MatrixXd Tt = T.transpose();               /* 物化，避免表达式临时对象 */
        Ti = Tt.partialPivLu().solve(Eigen::MatrixXd::Identity(n, n)).array().round();
        const double err = (Ti * Tt - Eigen::MatrixXd::Identity(n, n)).cwiseAbs().maxCoeff();
        const double det = std::fabs(T.determinant());
        r.T_mat  = T;
        r.Ti_mat = Ti;
        if (err > 0.5 || std::fabs(det - 1.0) > 1e-6) { if (out) *out = r; return false; }
    }

    /* 回原空间：a = Ti · z（Ti = T⁻ᵀ 是整数矩阵，乘法精确） */
    auto backToA = [&](const Eigen::VectorXd &z) {
        return (Ti * z).array().round().matrix();
    };

    /* ---- 1) 引导解（从后往前条件取整）给上界 ---- */
    Eigen::VectorXd zb = Eigen::VectorXd::Zero(n);
    for (int i = n - 1; i >= 0; --i) {
        double t = 0.0;
        for (int j = i + 1; j < n; ++j) t += R(i, j) * (zb(j) - at(j));
        zb(i) = std::round(at(i) - t / R(i, i));
    }
    const double sq_boot = quadFormR(R, zb, at);
    r.a_boot = backToA(zb);

    double p = 1.0;
    for (int i = 0; i < n; ++i) p *= (2.0 * normCdf(0.5 * R(i, i)) - 1.0);
    r.p_boot = p;

    /* ---- 2) 椭球枚举 ---- */
    /* 上界固定为 ratio_thresh · χ²_boot，不随 best 收缩：
     * 若区域内找不到第二个格点，则真次优 > ratio_thresh·χ²_boot ≥ ratio_thresh·χ²_best
     * ⇒ ratio > 门限 ⇒ 判可固定；这个推理是严格的，所以不会引入"假固定"。 */
    const double limit0 = ratio_thresh * sq_boot;
    /* 比较容差按问题尺度取（相对 sq_boot），保证 Q 与 c·Q 给出完全一致的结果 */
    const double tol_sq = 1e-12 * std::max(sq_boot, std::numeric_limits<double>::min());
    double best = sq_boot, second = std::numeric_limits<double>::infinity();
    Eigen::VectorXd zbest = zb, zsecond;
    long nodes = 0;
    bool overflow = false;
    Eigen::VectorXd z = zb;

    struct Frame { int i; double partial; int lo, hi, cur; };
    std::vector<Frame> st;

    auto pushFrame = [&](int i, double partial) {
        const double rem = limit0 - partial;
        if (rem <= 0.0 || i < 0) return;
        double t = 0.0;
        for (int j = i + 1; j < n; ++j) t += R(i, j) * (z(j) - at(j));
        const double center = at(i) - t / R(i, i);
        const double rad = std::sqrt(rem) / R(i, i);
        Frame f;
        f.i = i;
        f.partial = partial;
        f.lo = static_cast<int>(std::ceil(center - rad - 1e-9));
        f.hi = static_cast<int>(std::floor(center + rad + 1e-9));
        f.cur = f.lo;
        if (f.lo <= f.hi) st.push_back(f);
    };

    pushFrame(n - 1, 0.0);
    if (st.empty()) { if (out) *out = r; return false; }

    while (!st.empty()) {
        if (++nodes > max_nodes) { overflow = true; break; }
        Frame &fr = st.back();
        if (fr.cur > fr.hi) { st.pop_back(); continue; }
        const int i = fr.i;
        const int zi = fr.cur++;
        z(i) = zi;

        double t = 0.0;
        for (int j = i + 1; j < n; ++j) t += R(i, j) * (z(j) - at(j));
        const double s = R(i, i) * (zi - at(i)) + t;
        const double partial = fr.partial + s * s;

        if (i == 0) {
            if (partial < best - tol_sq) {
                second = best; zsecond = zbest;
                best = partial; zbest = z;
            } else if (partial < second - tol_sq) {
                bool same = true;
                for (int q = 0; q < n; ++q) if (std::fabs(z(q) - zbest(q)) > 0.5) { same = false; break; }
                if (!same) { second = partial; zsecond = z; }
            }
        } else {
            pushFrame(i - 1, partial);
        }
    }

    r.nodes = nodes;
    r.sq_best = best;
    r.sq_second = second;
    r.a_fixed = backToA(zbest);
    if (best <= 0.0) {
        /* 浮点解恰好落在整数格点上：最优即唯一，视为可固定 */
        r.ratio = std::numeric_limits<double>::infinity();
    } else {
        r.ratio = std::isfinite(second) ? second / best
                                        : std::numeric_limits<double>::infinity();
    }

    r.ratio_boot = sq_boot / best;   /* 引导解 vs 最优解 */

    /* ok 只表示“枚举正常完成”。ratio = inf 表示在 ratio_thresh·X2_boot 区域内
     * 找不到第二个格点 => 真次优 > 门限·X2_best => 可固定，这是**成功**而非失败
     * （早先要求 isfinite(ratio) 会把这种最好的情形误判成“未固定”）。 */
    r.ok = !overflow && std::isfinite(r.sq_best);
    if (out) *out = r;
    return r.ok;
}

bool ilsResolveBrute(const Eigen::VectorXd &a, const Eigen::MatrixXd &Q,
                     int range, IlsResult *out) {
    const int n = static_cast<int>(a.size());
    IlsResult r;
    r.n = n;
    r.a_float = a;
    if (n <= 0 || Q.rows() != n) { if (out) *out = r; return false; }

    Eigen::MatrixXd R;
    if (!cholInvUpper(Q, R)) { if (out) *out = r; return false; }

    std::vector<int> base(n);
    for (int i = 0; i < n; ++i) base[i] = static_cast<int>(std::lround(a(i)));

    double best = std::numeric_limits<double>::infinity();
    double second = std::numeric_limits<double>::infinity();
    Eigen::VectorXd z(n), zbest, zsecond;
    std::vector<int> off(n, 0);
    long nodes = 0;

    while (true) {
        ++nodes;
        for (int i = 0; i < n; ++i) z(i) = base[i] + off[i] - range;
        const double sq = quadFormR(R, z, a);
        if (sq < best - 1e-12) { second = best; zsecond = zbest; best = sq; zbest = z; }
        else if (sq < second - 1e-12) { second = sq; zsecond = z; }

        /* 标准里程计：off[k] ∈ [0, 2·range]，z = base + off − range。
         * （早先写成"到 range 就置 -range 并同时进位"，会漏掉一整片状态。） */
        int k = 0;
        while (k < n && off[k] == 2 * range) { off[k] = 0; ++k; }
        if (k >= n) break;
        ++off[k];
    }

    r.a_fixed = zbest;
    r.a_boot = zbest;
    r.sq_best = best;
    r.sq_second = second;
    r.ratio = std::isfinite(second) ? second / best : std::numeric_limits<double>::infinity();
    r.nodes = nodes;
    r.ok = std::isfinite(best);
    if (out) *out = r;
    return r.ok;
}


/* ============================================================================
 * 部分固定（partial AR）
 * ==========================================================================*/
bool ilsPartialFix(const Eigen::VectorXd &a, const Eigen::MatrixXd &Q,
                   double ratio_thresh, int min_fix, long max_nodes,
                   IlsPartial *out) {
    IlsPartial res;
    const int n = static_cast<int>(a.size());
    res.n_full = n;
    res.a_fixed = a;
    if (n <= 0 || Q.rows() != n) { if (out) *out = res; return false; }
    if (min_fix < 1) min_fix = 1;
    if (min_fix > n) min_fix = n;

    /* 条件标准差排序（小 = 有把握） */
    Eigen::MatrixXd L0;
    Eigen::VectorXd D0;
    if (!ldltUnit(Q, L0, D0)) { if (out) *out = res; return false; }
    res.cond_sigma.resize(n);
    for (int i = 0; i < n; ++i) res.cond_sigma(i) = std::sqrt(std::max(0.0, D0(i)));

    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int p, int q) { return res.cond_sigma(p) < res.cond_sigma(q); });

    /* 从"固定全部"往下退，取第一个能通过门限的子集 */
    for (int k = n; k >= min_fix; --k) {
        ++res.n_tried;
        std::vector<int> idx(order.begin(), order.begin() + k);
        std::sort(idx.begin(), idx.end());

        Eigen::VectorXd asub(k);
        Eigen::MatrixXd Qsub(k, k);
        for (int i = 0; i < k; ++i) {
            asub(i) = a(idx[static_cast<size_t>(i)]);
            for (int j = 0; j < k; ++j)
                Qsub(i, j) = Q(idx[static_cast<size_t>(i)], idx[static_cast<size_t>(j)]);
        }

        IlsResult r;
        if (!ilsResolve(asub, Qsub, &r, ratio_thresh, max_nodes)) continue;
        if (!(r.ratio > ratio_thresh)) continue;

        res.ok = true;
        res.fixed_idx = idx;
        res.last = r;
        for (int i = 0; i < k; ++i)
            res.a_fixed(idx[static_cast<size_t>(i)]) = r.a_fixed(i);
        if (out) *out = res;
        return true;
    }

    if (out) *out = res;
    return false;
}


/* ============================================================================
 * 以下为原 src/gnss_dd.h 的内容（RINEX → 双差 DDObs 生产者），已并入本头文件
 * ==========================================================================*/
/* ============================================================================
 * gnss_dd.cpp —— RINEX 3 观测/星历 → 双差 DDObs 的生产者
 * 符号约定见本文件顶部 gnss_dd 段（与 GnssDD*Residual 严格一致）。
 * ==========================================================================*/


/* ============================ 时间 ============================ */
/* Howard Hinnant 的 days_from_civil：自 1970-01-01 起的天数 */
static long daysFromCivil(int y, unsigned m, unsigned d) {
    y -= (m <= 2);
    const long era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153u * (m + (m > 2 ? -3u : 9u)) + 2u) / 5u + d - 1u;
    const unsigned doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
    return era * 146097L + static_cast<long>(doe) - 719468L;
}

double civilToGpsSec(int year, int month, int day, int hour, int minute, double sec) {
    static const long gps_epoch_day = daysFromCivil(1980, 1, 6);
    const long day_num = daysFromCivil(year, static_cast<unsigned>(month),
                                       static_cast<unsigned>(day));
    return static_cast<double>(day_num - gps_epoch_day) * 86400.0
         + hour * 3600.0 + minute * 60.0 + sec;
}

void gpsSecToCivil(double t_gps, int *year, int *month, int *day,
                   int *hour, int *minute, double *sec) {
    static const long gps_epoch_day = daysFromCivil(1980, 1, 6);
    const double fd = std::floor(t_gps / 86400.0);
    const long days = static_cast<long>(fd) + gps_epoch_day;
    double rem = t_gps - fd * 86400.0;

    const long z = days + 719468L;
    const long era = (z >= 0 ? z : z - 146096L) / 146097L;
    const unsigned doe = static_cast<unsigned>(z - era * 146097L);
    const unsigned yoe = (doe - doe / 1460u + doe / 36524u - doe / 146096u) / 365u;
    const int yy = static_cast<int>(yoe) + static_cast<int>(era) * 400;
    const unsigned doy = doe - (365u * yoe + yoe / 4u - yoe / 100u);
    const unsigned mp = (5u * doy + 2u) / 153u;
    const unsigned dd = doy - (153u * mp + 2u) / 5u + 1u;
    const unsigned mm = mp + (mp < 10u ? 3u : static_cast<unsigned>(-9));
    if (year)  *year  = yy + (mm <= 2u ? 1 : 0);
    if (month) *month = static_cast<int>(mm);
    if (day)   *day   = static_cast<int>(dd);
    const int hh = static_cast<int>(rem / 3600.0); rem -= hh * 3600.0;
    const int mi = static_cast<int>(rem / 60.0);   rem -= mi * 60.0;
    if (hour)   *hour   = hh;
    if (minute) *minute = mi;
    if (sec)    *sec    = rem;
}

/* ============================ 小工具 ============================ */
static inline std::string trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

/* RINEX 的 D 指数 → C 的 E 指数 */
static double parseD(const std::string &s) {
    std::string t = trim(s);
    if (t.empty()) return 0.0;
    for (char &c : t) if (c == 'D' || c == 'd') c = 'E';
    return std::strtod(t.c_str(), nullptr);
}

static double field19(const std::string &buf, size_t off) {
    if (off >= buf.size()) return 0.0;
    return parseD(buf.substr(off, 19));
}

/* ============================ 观测文件读取 ============================ */
bool readRinex3Obs(const std::string &path,
                   const std::vector<ObsRequest> &reqs,
                   std::vector<RinexEpoch> &epochs,
                   RinexObsHeader *hdr) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "[RINEX] 打不开观测文件: %s\n", path.c_str());
        return false;
    }

    RinexObsHeader h;
    std::string line;
    bool header_done = false;
    char cur_sys = 0;                 /* 头部续行沿用上一个系统 */

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() < 60) continue;
        const std::string label = trim(line.substr(60));
        if (label == "END OF HEADER") { header_done = true; break; }

        if (label == "RINEX VERSION / TYPE") {
            h.version = std::atof(line.substr(0, 9).c_str());
        } else if (label == "INTERVAL") {
            h.interval = std::atof(line.substr(0, 10).c_str());
        } else if (label == "APPROX POSITION XYZ") {
            h.approx_pos = Eigen::Vector3d(std::atof(line.substr(0, 14).c_str()),
                                           std::atof(line.substr(14, 14).c_str()),
                                           std::atof(line.substr(28, 14).c_str()));
            h.approx_pos_valid = h.approx_pos.norm() > 1.0e6;
        } else if (label == "MARKER NAME") {
            h.marker = trim(line.substr(0, 60));
        } else if (label == "TIME OF FIRST OBS") {
            int Y = 0, M = 0, D = 0, hh = 0, mm = 0; double ss = 0.0;
            std::sscanf(line.c_str(), "%d %d %d %d %d %lf", &Y, &M, &D, &hh, &mm, &ss);
            h.t_first = civilToGpsSec(Y, M, D, hh, mm, ss);
        } else if (label == "TIME OF LAST OBS") {
            int Y = 0, M = 0, D = 0, hh = 0, mm = 0; double ss = 0.0;
            std::sscanf(line.c_str(), "%d %d %d %d %d %lf", &Y, &M, &D, &hh, &mm, &ss);
            h.t_last = civilToGpsSec(Y, M, D, hh, mm, ss);
        } else if (label == "SYS / # / OBS TYPES") {
            if (std::isalpha(static_cast<unsigned char>(line[0]))) {
                cur_sys = line[0];
                const int n = std::atoi(line.substr(3, 3).c_str());
                std::vector<std::string> &vec = h.obs_types[cur_sys];
                vec.clear();
                vec.reserve(static_cast<size_t>(n));
                for (int k = 0; k < 13 && static_cast<int>(vec.size()) < n; ++k) {
                    const size_t off = 7 + 4 * static_cast<size_t>(k);
                    if (off + 3 > 60) break;
                    const std::string t = trim(line.substr(off, 3));
                    if (!t.empty()) vec.push_back(t);
                }
            } else if (cur_sys) {
                std::vector<std::string> &vec = h.obs_types[cur_sys];
                for (int k = 0; k < 13; ++k) {
                    const size_t off = 7 + 4 * static_cast<size_t>(k);
                    if (off + 3 > 60) break;
                    const std::string t = trim(line.substr(off, 3));
                    if (!t.empty()) vec.push_back(t);
                }
            }
        }
    }
    if (!header_done) {
        std::fprintf(stderr, "[RINEX] 头部没有 END OF HEADER: %s\n", path.c_str());
        return false;
    }

    /* ---- 请求 → 每系统的列索引 ---- */
    const int n_req = static_cast<int>(std::min<size_t>(reqs.size(), 12));
    std::array<char, 12> req_sys{};
    std::vector<int> req_col(reqs.size(), -1);
    for (int k = 0; k < n_req; ++k) {
        req_sys[static_cast<size_t>(k)] = reqs[static_cast<size_t>(k)].sys;
        auto it = h.obs_types.find(reqs[static_cast<size_t>(k)].sys);
        if (it == h.obs_types.end()) continue;
        for (size_t j = 0; j < it->second.size(); ++j) {
            if (it->second[j] == reqs[static_cast<size_t>(k)].type) {
                req_col[static_cast<size_t>(k)] = static_cast<int>(j);
                break;
            }
        }
    }

    /* ---- 历元 ---- */
    epochs.clear();
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] != '>') continue;

        int Y = 0, M = 0, D = 0, hh = 0, mm = 0, flag = 0, nsat = 0;
        double ss = 0.0, clk = 0.0;
        std::sscanf(line.c_str(), "> %d %d %d %d %d %lf %d %d %lf",
                    &Y, &M, &D, &hh, &mm, &ss, &flag, &nsat, &clk);

        RinexEpoch ep;
        ep.t = civilToGpsSec(Y, M, D, hh, mm, ss);
        ep.flag = flag;

        if (flag == 0 || flag == 1) {
            ep.sats.reserve(static_cast<size_t>(std::max(0, nsat)));
            for (int s = 0; s < nsat; ++s) {
                if (!std::getline(in, line)) break;
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.size() < 3) continue;
                const char sys = line[0];
                if (!std::isalpha(static_cast<unsigned char>(sys))) continue;

                RinexSatObs so;
                so.sys = sys;
                so.prn = std::atoi(line.substr(1, 2).c_str());
                bool any = false;
                for (int k = 0; k < n_req; ++k) {
                    if (req_sys[static_cast<size_t>(k)] != sys) continue;
                    const int col = req_col[static_cast<size_t>(k)];
                    if (col < 0) continue;
                    const size_t off = 3 + 16 * static_cast<size_t>(col);
                    if (off + 14 > line.size()) continue;
                    const std::string vstr = line.substr(off, 14);
                    if (trim(vstr).empty()) continue;
                    so.val[static_cast<size_t>(k)] = std::atof(vstr.c_str());
                    if (off + 14 < line.size()) {
                        const char lli = line[off + 14];
                        if (lli >= '0' && lli <= '9') so.lli[static_cast<size_t>(k)] = lli - '0';
                    }
                    if (off + 15 < line.size()) {
                        const char sn = line[off + 15];
                        if (sn >= '0' && sn <= '9') so.snr[static_cast<size_t>(k)] = sn - '0';
                    }
                    so.has[static_cast<size_t>(k)] = true;
                    any = true;
                }
                if (any) ep.sats.push_back(so);
            }
        }
        epochs.push_back(std::move(ep));
    }

    h.n_epochs = static_cast<int>(epochs.size());
    if (hdr) *hdr = h;
    if (epochs.empty()) {
        std::fprintf(stderr, "[RINEX] %s 没读到任何历元\n", path.c_str());
        return false;
    }
    return true;
}

/* ============================ 导航文件读取 ============================ */
static bool isNavRecordStart(const std::string &l) {
    if (l.size() < 4) return false;
    if (!std::isalpha(static_cast<unsigned char>(l[0]))) return false;
    if (!std::isdigit(static_cast<unsigned char>(l[1]))) return false;
    if (!std::isdigit(static_cast<unsigned char>(l[2]))) return false;
    return l[3] == ' ' || l[3] == '\t';
}

bool readRinex3Nav(const std::string &path,
                   std::map<char, std::map<int, std::vector<KeplerEph>>> &out,
                   int *n_glonass_skipped) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "[RINEX] 打不开导航文件: %s\n", path.c_str());
        return false;
    }
    std::string line;
    bool header_done = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() >= 60 && trim(line.substr(60)) == "END OF HEADER") {
            header_done = true;
            break;
        }
    }
    if (!header_done) {
        std::fprintf(stderr, "[RINEX] 导航文件没有 END OF HEADER: %s\n", path.c_str());
        return false;
    }

    int n_glo = 0;
    std::vector<std::string> rec;

    auto flush = [&]() {
        if (rec.empty()) return;
        const std::string &l0 = rec[0];
        if (l0.size() < 23) { rec.clear(); return; }
        const char sys = l0[0];
        const int prn = std::atoi(l0.substr(1, 2).c_str());
        if (sys == 'R') { ++n_glo; rec.clear(); return; }   /* GLONASS 状态向量，暂不支持 */
        if (sys == 'S') { rec.clear(); return; }            /* SBAS */

        int Y = 0, M = 0, D = 0, hh = 0, mm = 0; double ss = 0.0;
        std::sscanf(l0.c_str() + 3, "%d %d %d %d %d %lf", &Y, &M, &D, &hh, &mm, &ss);

        KeplerEph e;
        e.sys = sys;
        e.prn = prn;
        e.toc = civilToGpsSec(Y, M, D, hh, mm, ss);
        e.af0 = field19(l0, 23);
        e.af1 = field19(l0, 42);
        e.af2 = field19(l0, 61);

        /* 续行是"4 个空格 + 4×19.12"。注意字段起点是 4 + 19k，不是 19k ——
         * 若按 19k 取窗，会把数值的指数位截掉（例如 2.592000000000E+05 变成 E+0）。 */
        auto f = [&](size_t li, size_t fi) -> double {
            if (li >= rec.size()) return 0.0;
            return field19(rec[li], 4u + 19u * fi);
        };
        e.Crs       = f(1, 1);
        e.delta_n   = f(1, 2);
        e.M0        = f(1, 3);
        e.Cuc       = f(2, 0);
        e.e         = f(2, 1);
        e.Cus       = f(2, 2);
        e.sqrt_a    = f(2, 3);
        e.toe       = f(3, 0);
        e.Cic       = f(3, 1);
        e.OMEGA0    = f(3, 2);
        e.Cis       = f(3, 3);
        e.i0        = f(4, 0);
        e.Crc       = f(4, 1);
        e.omega     = f(4, 2);
        e.OMEGA_DOT = f(4, 3);
        e.idot      = f(5, 0);
        e.week      = static_cast<int>(f(5, 2));
        e.accuracy  = f(6, 0);
        e.health    = f(6, 1);
        e.tgd       = f(6, 2);

        e.valid = (e.sqrt_a > 1000.0 && e.toe > 0.0);
        if (e.valid) out[sys][prn].push_back(e);
        rec.clear();
    };

    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (isNavRecordStart(line)) {
            flush();
            rec.clear();
            rec.push_back(line);
        } else if (!rec.empty()) {
            rec.push_back(line);
        }
    }
    flush();

    if (n_glonass_skipped) *n_glonass_skipped = n_glo;

    size_t total = 0;
    for (auto &kv : out) for (auto &k2 : kv.second) total += k2.second.size();
    if (total == 0) {
        std::fprintf(stderr, "[RINEX] %s 没解析出任何星历\n", path.c_str());
        return false;
    }
    return true;
}

/* ============================ 卫星位置 ============================ */
static constexpr double OMEGA_E = 7.2921151467e-5;
static constexpr double GM_GPS  = 3.986005e14;
static constexpr double GM_BDS  = 3.986004418e14;
static constexpr double GM_GAL  = 3.986004418e14;

bool satPosFromEph(const KeplerEph &e, double t_tx,
                   Eigen::Vector3d &pos, double *clk) {
    if (!e.valid || e.sqrt_a < 1000.0) return false;

    const double GM = (e.sys == 'C') ? GM_BDS : ((e.sys == 'E') ? GM_GAL : GM_GPS);
    const double A  = e.sqrt_a * e.sqrt_a;
    const double n0 = std::sqrt(GM / (A * A * A));
    const double n  = n0 + e.delta_n;

    double tk = t_tx - e.toe;
    while (tk >  302400.0) tk -= 604800.0;
    while (tk < -302400.0) tk += 604800.0;

    const double M = e.M0 + n * tk;
    double E = M;
    for (int it = 0; it < 15; ++it) {
        const double dE = (E - e.e * std::sin(E) - M) / (1.0 - e.e * std::cos(E));
        E -= dE;
        if (std::fabs(dE) < 1e-13) break;
    }
    const double sinE = std::sin(E), cosE = std::cos(E);
    const double v = std::atan2(std::sqrt(1.0 - e.e * e.e) * sinE, cosE - e.e);
    const double phi = v + e.omega;
    const double s2 = std::sin(2.0 * phi), c2 = std::cos(2.0 * phi);

    const double u = phi + e.Cus * s2 + e.Cuc * c2;
    const double r = A * (1.0 - e.e * cosE) + e.Crs * s2 + e.Crc * c2;
    const double i = e.i0 + e.idot * tk + e.Cis * s2 + e.Cic * c2;

    const double xp = r * std::cos(u);
    const double yp = r * std::sin(u);

    if (e.sys == 'C' && e.prn <= 5) {
        /* BDS GEO：ICD 的另一套旋转 */
        const double Om  = e.OMEGA0 + e.OMEGA_DOT * tk - OMEGA_E * e.toe;
        const double a1  =  std::sin(Om) * xp + std::cos(Om) * std::cos(i) * yp;
        const double b1  = -std::cos(Om) * xp + std::sin(Om) * std::cos(i) * yp;
        const double c1  =  std::sin(i) * yp;
        const double Omk = Om - OMEGA_E * tk;
        const double c5  = std::cos(-5.0 * DEG), s5 = std::sin(-5.0 * DEG);
        pos.x() =  a1 * std::cos(Omk) + b1 * std::sin(Omk) * c5 + c1 * std::sin(Omk) * s5;
        pos.y() = -a1 * std::sin(Omk) + b1 * std::cos(Omk) * c5 + c1 * std::cos(Omk) * s5;
        pos.z() = -b1 * s5 + c1 * c5;
    } else {
        const double Om = e.OMEGA0 + (e.OMEGA_DOT - OMEGA_E) * tk - OMEGA_E * e.toe;
        pos.x() = xp * std::cos(Om) - yp * std::cos(i) * std::sin(Om);
        pos.y() = xp * std::sin(Om) + yp * std::cos(i) * std::cos(Om);
        pos.z() = yp * std::sin(i);
    }

    if (clk) {
        double dts = e.af0 + e.af1 * tk + e.af2 * tk * tk;
        dts -= 2.0 * std::sqrt(GM * A) / (CLIGHT * CLIGHT) * e.e * sinE;
        *clk = dts;
    }
    return true;
}

/* 迭代求发射时刻（考虑传播时延与地球自转） */
static bool satPosAtRx(const KeplerEph &e, double t_rx, const Eigen::Vector3d &rx,
                       Eigen::Vector3d &pos, double *clk) {
    double tau = 0.07;
    double dt_clk = 0.0;
    for (int it = 0; it < 3; ++it) {
        if (!satPosFromEph(e, t_rx - tau, pos, clk)) return false;
        dt_clk = clk ? *clk : 0.0;
        tau = (pos - rx).norm() / CLIGHT;
    }
    /* 地球自转改正：信号传播期间地球转了 ω·τ */
    const double theta = OMEGA_E * tau;
    const double ct = std::cos(theta), st = std::sin(theta);
    const double x =  ct * pos.x() + st * pos.y();
    const double y = -st * pos.x() + ct * pos.y();
    pos.x() = x;
    pos.y() = y;
    if (clk) *clk = dt_clk;
    return true;
}

/* ============================ 请求表 ============================ */
std::vector<ObsRequest> buildObsRequests(const std::vector<DdSignal> &slots,
                                         std::vector<std::array<int, 3>> *slot_index) {
    std::vector<ObsRequest> reqs;
    if (slot_index) slot_index->clear();
    const size_t n = std::min<size_t>(slots.size(), MAX_FREQ);
    for (size_t s = 0; s < n; ++s) {
        std::array<int, 3> idx{-1, -1, -1};
        const char *codes[3] = {slots[s].code_P.c_str(), slots[s].code_L.c_str(),
                                slots[s].code_D.c_str()};
        for (int j = 0; j < 3; ++j) {
            if (codes[j][0] == '\0') continue;
            idx[static_cast<size_t>(j)] = static_cast<int>(reqs.size());
            ObsRequest r;
            r.sys  = slots[s].sys;
            r.type = codes[j];
            reqs.push_back(r);
        }
        if (slot_index) slot_index->push_back(idx);
    }
    return reqs;
}

/* ============================ 双差生产 ============================ */
namespace {
struct SatKey {
    char sys;
    int  prn;
    bool operator<(const SatKey &o) const {
        return (sys != o.sys) ? (sys < o.sys) : (prn < o.prn);
    }
};
struct Cand {
    SatKey key;
    const RinexSatObs *rover{nullptr};
    const RinexSatObs *base{nullptr};
    Eigen::Vector3d pos;
    double clk{0.0};
    double elev{0.0};
};
}  /* namespace */

/* ============================ MW 周跳探测 ============================ */
bool ddMelbourneWubbena(const DDObs &dd, int f1, int f2, double *n_wl) {
    if (f1 < 0 || f2 < 0 || f1 >= MAX_FREQ || f2 >= MAX_FREQ || f1 == f2) return false;
    if (!dd.has_P[f1] || !dd.has_L[f1] || !dd.has_P[f2] || !dd.has_L[f2]) return false;
    const double fr1 = GPS_FREQ[f1].freq_hz, fr2 = GPS_FREQ[f2].freq_hz;
    const double lam_wl = CLIGHT / (fr1 - fr2);

    /* 宽巷相位（周） */
    const double L_wl = (fr1 * dd.dd_L[f1] - fr2 * dd.dd_L[f2]) / (fr1 - fr2);
    /* 窄巷伪距（m） */
    const double P_nl = (fr1 * dd.dd_P[f1] + fr2 * dd.dd_P[f2]) / (fr1 + fr2);

    if (n_wl) *n_wl = L_wl - P_nl / lam_wl;
    return true;
}

bool buildDdEpochs(const std::vector<RinexEpoch> &rover,
                   const std::vector<RinexEpoch> &base,
                   const std::map<char, std::map<int, std::vector<KeplerEph>>> &eph,
                   const DdBuildConfig &cfg,
                   std::vector<DdEpochResult> &out,
                   DdBuildStats *stats) {
    out.clear();
    DdBuildStats st_local;
    if (stats) *stats = st_local;
    if (rover.empty() || base.empty()) return false;
    if (cfg.slots.empty() || cfg.slots.size() > MAX_FREQ) return false;

    std::vector<std::array<int, 3>> slot_index;
    buildObsRequests(cfg.slots, &slot_index);
    const int n_slot = static_cast<int>(slot_index.size());

    /* MW 周跳状态：按 (ref, sat, slot) 键维护最近若干历元的 N_WL（滑动窗口） */
    struct MwState { std::vector<double> buf; };
    std::map<std::pair<AmbKey, int>, MwState> mw_state;
    int n_slip_total = 0;

    std::vector<double> bt(base.size());
    for (size_t i = 0; i < base.size(); ++i) bt[i] = base[i].t;
    const bool bt_sorted = std::is_sorted(bt.begin(), bt.end());

    const double elev_mask = cfg.elev_mask_deg * DEG;
    const Eigen::Vector3d rover_ref = cfg.rover_approx;

    /* ================= 第 1 遍：逐历元收集候选卫星 ================= */
    struct EpochCands { double t; std::vector<Cand> cands; };
    std::vector<EpochCands> ec;

    for (const RinexEpoch &re : rover) {
        ++st_local.n_epochs_rover;
        if (re.flag != 0 && re.flag != 1) continue;

        const RinexEpoch *be = nullptr;
        if (bt_sorted) {
            auto it = std::lower_bound(bt.begin(), bt.end(), re.t);
            size_t idx = static_cast<size_t>(it - bt.begin());
            if (idx > 0 && (idx >= base.size() ||
                            std::fabs(bt[idx - 1] - re.t) <= std::fabs(bt[idx] - re.t)))
                --idx;
            if (idx < base.size() && std::fabs(bt[idx] - re.t) <= cfg.max_base_dt)
                be = &base[idx];
        }
        if (!be) { ++st_local.n_no_base; continue; }

        std::vector<Cand> cands;
        for (const RinexSatObs &rs : re.sats) {
            const RinexSatObs *bs = nullptr;
            for (const RinexSatObs &b : be->sats) {
                if (b.sys == rs.sys && b.prn == rs.prn) { bs = &b; break; }
            }
            if (!bs) continue;

            bool all = true;
            for (int s = 0; s < n_slot && all; ++s) {
                const int iP = slot_index[static_cast<size_t>(s)][0];
                const int iL = slot_index[static_cast<size_t>(s)][1];
                if (iP >= 0 && (!rs.ok(iP) || !bs->ok(iP))) all = false;
                if (iL >= 0 && (!rs.ok(iL) || !bs->ok(iL))) all = false;
            }
            if (!all) continue;

            auto es = eph.find(rs.sys);
            if (es == eph.end()) continue;
            auto ep = es->second.find(rs.prn);
            if (ep == es->second.end() || ep->second.empty()) continue;

            const KeplerEph *best_e = nullptr;
            double best_dt = 1e18;
            for (const KeplerEph &cand : ep->second) {
                const double d = std::fabs(cand.toe - re.t);
                if (d < best_dt) { best_dt = d; best_e = &cand; }
            }
            if (!best_e) continue;

            Cand cd;
            cd.key = SatKey{rs.sys, rs.prn};
            cd.rover = &rs;
            cd.base = bs;
            if (!satPosAtRx(*best_e, re.t, rover_ref, cd.pos, &cd.clk)) continue;

            const Eigen::Vector3d los = cd.pos - rover_ref;
            const double rng = los.norm();
            if (rng < 1.0e6) continue;
            const Eigen::Vector3d up = rover_ref.normalized();
            cd.elev = std::asin(std::max(-1.0, std::min(1.0, los.dot(up) / rng)));
            if (cd.elev < elev_mask) continue;

            cands.push_back(cd);
        }

        if (static_cast<int>(cands.size()) < cfg.min_sats) { ++st_local.n_too_few; continue; }
        ec.push_back(EpochCands{re.t, std::move(cands)});
    }

    /* ================= 选参考星（每系统一颗，全窗口固定） ================= */
    std::map<char, int> chosen;                     /* 系统 → PRN */
    if (cfg.stable_reference) {
        std::map<std::pair<char, int>, int> cnt;    /* (系统,PRN) → 可用历元数 */
        std::map<std::pair<char, int>, double> elev_sum;
        for (const EpochCands &e : ec) {
            for (const Cand &c : e.cands) {
                ++cnt[{c.key.sys, c.key.prn}];
                elev_sum[{c.key.sys, c.key.prn}] += c.elev;
            }
        }
        for (const auto &kv : cnt) {
            const char sys = kv.first.first;
            const int  prn = kv.first.second;
            auto it = chosen.find(sys);
            if (it == chosen.end()) { chosen[sys] = prn; continue; }
            const int cur = it->second;
            const int c_new = cnt[{sys, prn}];
            const int c_cur = cnt[{sys, cur}];
            if (c_new > c_cur) { chosen[sys] = prn; continue; }
            if (c_new == c_cur) {
                const double e_new = elev_sum[{sys, prn}] / c_new;
                const double e_cur = elev_sum[{sys, cur}] / c_cur;
                if (e_new > e_cur) chosen[sys] = prn;
            }
        }
    }

    /* ================= 第 2 遍：按选定参考星组双差 ================= */
    for (const EpochCands &e : ec) {
        std::map<char, std::vector<const Cand *>> by_sys;
        for (const Cand &c : e.cands) by_sys[c.key.sys].push_back(&c);

        DdEpochResult res;
        res.t = e.t;
        res.n_common = static_cast<int>(e.cands.size());

        for (auto &kv : by_sys) {
            const std::vector<const Cand *> &g = kv.second;
            if (g.size() < 2) continue;

            size_t iref = static_cast<size_t>(-1);
            if (cfg.stable_reference) {
                auto it = chosen.find(kv.first);
                if (it != chosen.end()) {
                    for (size_t i = 0; i < g.size(); ++i) {
                        if (g[i]->key.prn == it->second) { iref = i; break; }
                    }
                }
                if (iref == static_cast<size_t>(-1)) {
                    /* 选定参考星本历元不可用：不用别的星顶替（会引入新模糊度参数） */
                    ++st_local.n_ref_missing;
                    continue;
                }
            } else {
                iref = 0;
                for (size_t i = 1; i < g.size(); ++i)
                    if (g[i]->elev > g[iref]->elev) iref = i;
            }

            const Cand &ref = *g[iref];
            if (res.ref_prn == 0) res.ref_prn = ref.key.prn;
            bool sys_used = false;

            for (size_t i = 0; i < g.size(); ++i) {
                if (i == iref) continue;
                const Cand &s = *g[i];

                DDObs dd;
                dd.ref_sat = ref.key.prn;
                dd.sat     = s.key.prn;
                std::array<bool, MAX_FREQ> slip{};
                slip.fill(false);

                for (int sl = 0; sl < n_slot; ++sl) {
                    const int iP = slot_index[static_cast<size_t>(sl)][0];
                    const int iL = slot_index[static_cast<size_t>(sl)][1];
                    const int iD = slot_index[static_cast<size_t>(sl)][2];

                    if (iP >= 0) {
                        dd.dd_P[sl] = (s.rover->val[static_cast<size_t>(iP)] -
                                       ref.rover->val[static_cast<size_t>(iP)])
                                    - (s.base ->val[static_cast<size_t>(iP)] -
                                       ref.base ->val[static_cast<size_t>(iP)]);
                        dd.has_P[sl] = true;
                    }
                    if (iL >= 0) {
                        dd.dd_L[sl] = (s.rover->val[static_cast<size_t>(iL)] -
                                       ref.rover->val[static_cast<size_t>(iL)])
                                    - (s.base ->val[static_cast<size_t>(iL)] -
                                       ref.base ->val[static_cast<size_t>(iL)]);
                        dd.has_L[sl] = true;
                    }
                    if (iD >= 0 && s.rover->ok(iD) && ref.rover->ok(iD) &&
                        s.base->ok(iD) && ref.base->ok(iD)) {
                        dd.dd_D[sl] = (s.rover->val[static_cast<size_t>(iD)] -
                                       ref.rover->val[static_cast<size_t>(iD)])
                                    - (s.base ->val[static_cast<size_t>(iD)] -
                                       ref.base ->val[static_cast<size_t>(iD)]);
                        dd.has_D[sl] = true;
                    }
                    if (cfg.use_lli_slip) {
                        const int idx = (iP >= 0) ? iP : iL;
                        const int lli[4] = {s.rover->lli[static_cast<size_t>(idx)],
                                            ref.rover->lli[static_cast<size_t>(idx)],
                                            s.base ->lli[static_cast<size_t>(idx)],
                                            ref.base ->lli[static_cast<size_t>(idx)]};
                        for (int q = 0; q < 4; ++q) if (lli[q] & 0x1) slip[sl] = true;
                    }
                }

                /* MW 周跳：几何无关 + 电离层无关。用**弧段均值**判偏差，
                 * 而不是与上一历元相减 —— 后者的噪声（≈0.5 周）会把门限淹掉。 */
                if (cfg.mw_slip_thresh > 0.0 && n_slot >= 2) {
                    double mw = std::numeric_limits<double>::quiet_NaN();
                    if (ddMelbourneWubbena(dd, 0, 1, &mw) && std::isfinite(mw)) {
                        for (int sl = 0; sl < n_slot; ++sl) {
                            const AmbKey base_key = ambKey(dd.ref_sat, dd.sat, sl);
                            const auto key = std::make_pair(base_key, sl);
                            auto it = mw_state.find(key);
                            if (it == mw_state.end()) {
                                MwState ns;
                                ns.buf.push_back(mw);
                                mw_state[key] = std::move(ns);
                                continue;
                            }
                            MwState &stt = it->second;
                            double mean = 0.0;
                            for (double v : stt.buf) mean += v;
                            mean /= static_cast<double>(stt.buf.size());
                            if (static_cast<int>(stt.buf.size()) >= cfg.mw_min_samples &&
                                std::fabs(mw - mean) > cfg.mw_slip_thresh) {
                                slip[sl] = true;
                                stt.buf.assign(1, mw);      /* 跳变后重新起窗 */
                                continue;
                            }
                            stt.buf.push_back(mw);
                            if (static_cast<int>(stt.buf.size()) > cfg.mw_window)
                                stt.buf.erase(stt.buf.begin());
                        }
                    }
                }
                for (int sl = 0; sl < n_slot; ++sl) if (slip[sl]) ++n_slip_total;

                dd.sigma_P = GPS_FREQ[0].sigma_P * std::sqrt(2.0);
                dd.sigma_L = GPS_FREQ[0].sigma_L;

                auto fillSat = [](SatState &stt, const Cand &c) {
                    stt.sat_id     = c.key.prn;
                    stt.pos_ecef   = c.pos;
                    stt.clk_bias_s = c.clk;
                    stt.elev_rad   = c.elev;
                    stt.valid      = true;
                };
                fillSat(dd.sat_i, s);
                fillSat(dd.sat_k, ref);

                res.dd.push_back(dd);
                res.slip.push_back(slip);
                sys_used = true;
            }
            if (sys_used) ++res.n_sys;
        }

        if (res.dd.empty()) continue;
        res.n_used = static_cast<int>(res.dd.size());
        out.push_back(std::move(res));
    }

    st_local.n_epochs_out = static_cast<int>(out.size());
    st_local.n_slip = n_slip_total;
    st_local.chosen_ref = chosen;
    if (stats) *stats = st_local;
    return !out.empty();
}

/* ============================ 挂到 Frame ============================ */
int attachDdToFrames(std::vector<Frame> &frames,
                     const std::vector<DdEpochResult> &dd_epochs,
                     double t_offset, double tol) {
    if (frames.empty() || dd_epochs.empty()) return 0;

    /* 为**全部**双差历元按 (ref_sat, sat, slot) 预计算弧段号。
     * 关键点：必须在全历元序列上累计计数器，而不是在"匹配到的帧"上累计；
     * 关键帧采样率通常低于 GNSS 采样率，逐帧累计会漏掉未匹配历元里的周跳，
     * 从而把两条不同的弧串成同一个键。 */
    std::vector<std::vector<std::array<int, MAX_FREQ>>> dd_arc_all(dd_epochs.size());
    {
        std::map<AmbKey, int> arc_of;
        for (size_t e = 0; e < dd_epochs.size(); ++e) {
            const DdEpochResult &ep = dd_epochs[e];
            dd_arc_all[e].assign(ep.dd.size(), {0, 0, 0});
            for (size_t i = 0; i < ep.dd.size(); ++i) {
                for (int slot = 0; slot < MAX_FREQ; ++slot) {
                    const AmbKey ak = ambKey(ep.dd[i].ref_sat, ep.dd[i].sat, slot);
                    if (ep.slip[i][slot]) ++arc_of[ak];
                    dd_arc_all[e][i][slot] = arc_of[ak];
                }
            }
        }
    }

    std::vector<double> dt(dd_epochs.size());
    for (size_t i = 0; i < dd_epochs.size(); ++i) dt[i] = dd_epochs[i].t;

    int n_ok = 0;
    for (Frame &f : frames) {
        const double want = f.time + t_offset;
        auto it = std::lower_bound(dt.begin(), dt.end(), want);
        size_t idx = static_cast<size_t>(it - dt.begin());
        if (idx > 0 && (idx >= dd_epochs.size() ||
                        std::fabs(dt[idx - 1] - want) <= std::fabs(dt[idx] - want)))
            --idx;
        if (idx >= dd_epochs.size()) continue;
        if (std::fabs(dt[idx] - want) > tol) continue;

        f.dd_obs = dd_epochs[idx].dd;
        f.dd_arc = dd_arc_all[idx];
        f.has_gnss_raw = !f.dd_obs.empty();
        if (f.has_gnss_raw) ++n_ok;
    }
    return n_ok;
}

