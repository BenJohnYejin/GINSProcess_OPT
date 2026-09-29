/**
 * @file alg_lib.cpp
 * @brief 算法库实现
 *
 * 分节顺序与 alg_lib.h 的目录一致（自上而下）：
 *   1. 基础类型           vect3 / mat3 / quat / vect / mat
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
vect3 O31(0.0);
vect3 One31(1.0);
mat3 I33(1, 0, 0, 0, 1, 0, 0, 0, 1);
quat qI(1, 0, 0, 0);

/* ============================================================================
 * vect3
 * ==========================================================================*/
int vect3::IsZeros(const vect3 &v, double eps) const { return ::IsZeros(v, eps); }

bool vect3::IsZeroXY(const vect3 &v, double eps) const {
    return (v.i < eps && v.i > -eps && v.j < eps && v.j > -eps);
}

bool vect3::IsNaN(const vect3 &v) const {
    return std::isnan(v.i) || std::isnan(v.j) || std::isnan(v.k);
}

vect3 &vect3::operator=(double f) {
    i = j = k = f;
    return *this;
}

vect3 &vect3::operator=(const double *pf) {
    i = pf[0];
    j = pf[1];
    k = pf[2];
    return *this;
}

vect3 vect3::operator+(const vect3 &v) const { return vect3(toEigen() + v.toEigen()); }

vect3 vect3::operator-(const vect3 &v) const { return vect3(toEigen() - v.toEigen()); }

vect3 vect3::operator*(const vect3 &v) const { return vect3(toEigen().cross(v.toEigen())); }

vect3 vect3::operator*(double f) const { return vect3(toEigen() * f); }

vect3 vect3::operator*(const mat3 &m) const {
    /* 与 ipos3g 一致：3x1 视作 1x3 行向量后右乘矩阵，即 v^T * M */
    Eigen::RowVector3d r = toEigen().transpose() * m.e();
    return vect3(r(0), r(1), r(2));
}

vect3 vect3::operator/(double f) const { return vect3(toEigen() / f); }

vect3 vect3::operator/(const vect3 &v) const {
    return vect3(toEigen().cwiseQuotient(v.toEigen()));
}

vect3 &vect3::operator+=(const vect3 &v) {
    e() += v.toEigen();
    return *this;
}

vect3 &vect3::operator-=(const vect3 &v) {
    e() -= v.toEigen();
    return *this;
}

vect3 &vect3::operator*=(double f) {
    e() *= f;
    return *this;
}

vect3 &vect3::operator/=(double f) {
    e() /= f;
    return *this;
}

vect3 &vect3::operator/=(const vect3 &v) {
    e() = e().cwiseQuotient(v.toEigen());
    return *this;
}

vect3 operator*(double f, const vect3 &v) { return vect3(v.toEigen() * f); }

vect3 operator-(const vect3 &v) { return vect3(-v.toEigen()); }

/* ============================================================================
 * mat3
 * ==========================================================================*/

/* 从 double / float 数组按行主序构造 ----------------------------------------------*/
mat3::mat3(const double *pxyz) {
    set(pxyz[0], pxyz[1], pxyz[2], pxyz[3], pxyz[4], pxyz[5], pxyz[6], pxyz[7], pxyz[8]);
}

mat3::mat3(const float *pxyz) {
    set(pxyz[0], pxyz[1], pxyz[2], pxyz[3], pxyz[4], pxyz[5], pxyz[6], pxyz[7], pxyz[8]);
}

/* 由三个矢量构造：isrow=1 时按行，否则按列 -----------------------------------------*/
mat3::mat3(const vect3 &v0, const vect3 &v1, const vect3 &v2, bool isrow) {
    if (isrow) {
        set(v0.i, v0.j, v0.k, v1.i, v1.j, v1.k, v2.i, v2.j, v2.k);
    } else {
        set(v0.i, v1.i, v2.i, v0.j, v1.j, v2.j, v0.k, v1.k, v2.k);
    }
}

mat3::mat3(const Eigen::Matrix3d &m) {
    set(m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2), m(2, 0), m(2, 1), m(2, 2));
}

mat3 mat3::operator+(const mat3 &m) const { return mat3(e() + m.e()); }

mat3 mat3::operator-(const mat3 &m) const { return mat3(e() - m.e()); }

mat3 mat3::operator*(const mat3 &m) const { return mat3(e() * m.e()); }

mat3 mat3::operator*(double f) const { return mat3(e() * f); }

vect3 mat3::operator*(const vect3 &v) const { return vect3(e() * v.toEigen()); }

mat3 &mat3::operator+=(const mat3 &m) {
    e() += m.e();
    return *this;
}

/* v 加到对角线 -------------------------------------------------------------------*/
mat3 mat3::operator+(const vect3 &v) const {
    mat3 mtmp = *this;
    mtmp.e00 += v.i;
    mtmp.e11 += v.j;
    mtmp.e22 += v.k;
    return mtmp;
}

mat3 &mat3::operator+=(const vect3 &v) {
    e00 += v.i;
    e11 += v.j;
    e22 += v.k;
    return *this;
}

void mat3::SetRow(int i, const vect3 &v) {
    double *p = &e00 + i * 3;
    p[0] = v.i;
    p[1] = v.j;
    p[2] = v.k;
}

void mat3::SetClm(int i, const vect3 &v) {
    double *p = &e00 + i;
    p[0] = v.i;
    p[3] = v.j;
    p[6] = v.k;
}

vect3 mat3::GetRow(int i) const {
    const double *p = &e00 + i * 3;
    return vect3(p[0], p[1], p[2]);
}

vect3 mat3::GetClm(int i) const {
    const double *p = &e00 + i;
    return vect3(p[0], p[3], p[6]);
}

mat3 operator-(const mat3 &m) { return mat3(-m.e()); }

mat3 operator~(const mat3 &m) { return mat3(m.e().transpose()); }

mat3 operator*(double f, const mat3 &m) { return mat3(m.e() * f); }

double &mat3::operator()(int i, int j ) {
    int r = i, c = (j < 0) ? i : j;
    switch (r * 3 + c) {
        case 0: return e00; case 1: return e01; case 2: return e02;
        case 3: return e10; case 4: return e11; case 5: return e12;
        case 6: return e20; case 7: return e21; case 8: return e22;
        default: return e00;
    }
}
double mat3::operator()(int i, int j ) const {
    int r = i, c = (j < 0) ? i : j;
    switch (r * 3 + c) {
        case 0: return e00; case 1: return e01; case 2: return e02;
        case 3: return e10; case 4: return e11; case 5: return e12;
        case 6: return e20; case 7: return e21; case 8: return e22;
        default: return e00;
    }
}

/* ============================================================================
 * quat
 * ==========================================================================*/

quat quat::operator+(const vect3 &phi) const { return rv2q(-phi) * (*this); }

quat quat::operator-(const vect3 &phi) const { return rv2q(phi) * (*this); }

/* 失准角：与 ipos3g 一致，dq = quat * ~(*this) -----------------------------------*/
vect3 quat::operator-(const quat &quat0) const {
    quat dq = quat0 * (~(*this));
    if (dq.q0 < 0) {
        dq.q0 = -dq.q0;
        dq.q1 = -dq.q1;
        dq.q2 = -dq.q2;
        dq.q3 = -dq.q3;
    }
    double n2 = std::acos(range(dq.q0, -1.0, 1.0)), f;
    if (sign(n2) != 0) {
        f = 2.0 / (std::sin(n2) / n2);
    } else {
        f = 2.0;
    }
    return vect3(dq.q1, dq.q2, dq.q3) * f;
}

quat quat::operator*(const quat &quat0) const { return quat(toEigen() * quat0.toEigen()); }

quat &quat::operator*=(const quat &quat0) {
    *this = *this * quat0;
    return *this;
}

quat &quat::operator-=(const vect3 &phi) {
    *this = rv2q(phi) * (*this);
    return *this;
}

quat operator~(const quat &q) { return quat(q.q0, -q.q1, -q.q2, -q.q3); }

/* 用四元数旋转矢量：Cnb * v --------------------------------------------------------*/
vect3 quat::operator*(const vect3 &v) const { return vect3(toEigen() * v.toEigen()); }

void quat::SetYaw(double yaw) {
    vect3 att = q2att(*this);
    att.k     = yaw;
    *this     = a2qua(att);
}

void normlize(quat *q) {
    double nq = std::sqrt(q->q0 * q->q0 + q->q1 * q->q1 + q->q2 * q->q2 + q->q3 * q->q3);
    q->q0 /= nq;
    q->q1 /= nq;
    q->q2 /= nq;
    q->q3 /= nq;
}



void vect::resize(int r, int c) {
    row = r;
    clm = c;
    rc  = r * c;
    if (rc > 0) {
        E.resize(r, c);
    } else {
        /* 保持 dd 合法，避免空对象出现悬垂指针 */
        E.resize(1, 1);
    }
    bind();
}

vect::vect(void) : row(0), clm(0), rc(0), dd(nullptr) { resize(0, 0); }

vect::vect(int row0, int clm0) : dd(nullptr) {
    /* 与 ipos3g 一致：clm0==1 时是列向量，否则退化为行向量 */
    if (clm0 == 1) {
        resize(row0, 1);
    } else {
        resize(1, clm0);
    }
    E.setZero();
}

vect::vect(int row0, double f) : dd(nullptr) {
    resize(row0, 1);
    E.setConstant(f);
}

vect::vect(int row0, double f, double f1, ...) : dd(nullptr) {
    resize(row0, 1);
    E.setZero();
    dd[0] = f;
    va_list vl;
    va_start(vl, f1);
    double cur = f1;
    for (int i = 1; i < rc; i++) {
        if (isInfSentinel(cur)) {
            break;
        }
        dd[i] = cur;
        if (i + 1 < rc) {
            cur = va_arg(vl, double);
        }
    }
    va_end(vl);
}

vect::vect(int row0, const double *pf) : dd(nullptr) {
    resize(row0, 1);
    std::memcpy(dd, pf, rc * sizeof(double));
}

vect::vect(const vect3 &v) : dd(nullptr) {
    resize(3, 1);
    dd[0] = v.i;
    dd[1] = v.j;
    dd[2] = v.k;
}

vect::vect(const vect3 &v1, const vect3 v2) : dd(nullptr) {
    resize(6, 1);
    dd[0] = v1.i;
    dd[1] = v1.j;
    dd[2] = v1.k;
    dd[3] = v2.i;
    dd[4] = v2.j;
    dd[5] = v2.k;
}

vect::vect(const Eigen::VectorXd &v) : dd(nullptr) {
    resize(static_cast<int>(v.size()), 1);
    std::memcpy(dd, v.data(), rc * sizeof(double));
}

vect::vect(const vect &o) : row(o.row), clm(o.clm), rc(o.rc), E(o.E), dd(nullptr) { bind(); }

vect &vect::operator=(const vect &o) {
    if (this == &o) {
        return *this;
    }
    row = o.row;
    clm = o.clm;
    rc  = o.rc;
    E   = o.E;
    if (rc == 0 && E.size() == 0) {
        E.resize(1, 1);
    }
    bind();
    return *this;
}

void vect::Set(double f[], int size) {
    int j = 0;
    for (int i = 0; i < rc && j < size; i++, j++) {
        if (isInfSentinel(f[j])) {
            break;
        }
        dd[i] = f[j];
    }
}

void vect::Set2(double f[], int size) {
    int j = 0;
    for (int i = 0; i < rc && j < size; i++, j++) {
        if (isInfSentinel(f[j])) {
            break;
        }
        dd[i] = f[j] * f[j];
    }
}

void vect::SetVect3(int i, const vect3 &v) {
    dd[i]     = v.i;
    dd[i + 1] = v.j;
    dd[i + 2] = v.k;
}

void vect::Set2Vect3(int i, const vect3 &v) {
    dd[i++] = v.i * v.i;
    dd[i++] = v.j * v.j;
    dd[i]   = v.k * v.k;
}

void vect::SetBit(unsigned int bit, double f) {
    for (int i = 0; i < rc; i++) {
        if (bit & (0x01u << i)) {
            dd[i] = f;
        }
    }
}

void vect::SetBit(unsigned int bit, const vect3 &v) {
    const double *p = &v.i;
    for (int i = 0; i < rc; i++) {
        if (bit & (0x01u << i)) {
            dd[i] = *p++;
            if (p > &v.k) {
                p = &v.i;
            }
        }
    }
}

vect3 vect::GetVect3(int i) const { return vect3(dd[i], dd[i + 1], dd[i + 2]); }

vect vect::operator+(const vect &v) const {
    assert(row == v.row && clm == v.clm);
    vect vtmp(row, clm);
    vtmp.e() = e() + v.e();
    return vtmp;
}

vect vect::operator-(const vect &v) const {
    assert(row == v.row && clm == v.clm);
    vect vtmp(row, clm);
    vtmp.e() = e() - v.e();
    return vtmp;
}

vect vect::operator*(double f) const {
    vect vtmp(row, clm);
    vtmp.e() = e() * f;
    return vtmp;
}

vect &vect::operator=(double f) {
    E.setConstant(f);
    return *this;
}

vect &vect::operator=(const double *pf) {
    std::memcpy(dd, pf, rc * sizeof(double));
    return *this;
}

vect &vect::operator=(const mat3 &m) {
    resize(9, 1);
    std::memcpy(dd, &m.e00, 9 * sizeof(double));
    return *this;
}

vect &vect::operator+=(const vect &v) {
    assert(row == v.row && clm == v.clm);
    e() += v.e();
    return *this;
}

vect &vect::operator-=(const vect &v) {
    assert(row == v.row && clm == v.clm);
    e() -= v.e();
    return *this;
}

vect &vect::operator*=(double f) {
    e() *= f;
    return *this;
}

/* 外积：v(1xn) * w(nx1) -> (1x1)；v(nx1) * w(1xm) -> (nxm) ----------------------*/
mat vect::operator*(const vect &v) const {
    assert(clm == v.row);
    if (row == 1 && v.clm == 1) {
        mat mtmp(1, 1);
        mtmp.dd[0] = dot(*this, v);
        return mtmp;
    }
    mat mtmp(row, v.clm);
    mtmp.e() = e() * v.e();
    return mtmp;
}

vect operator~(const vect &v) {
    vect vtmp = v;
    /* 数据保持不变，仅交换形状（rc 不变） */
    vtmp.row = v.clm;
    vtmp.clm = v.row;
    return vtmp;
}

/* ============================================================================
 * mat
 * ==========================================================================*/
void mat::resize(int r, int c) {
    row = r;
    clm = c;
    rc  = r * c;
    if (rc > 0) {
        E.resize(r, c);
    } else {
        E.resize(1, 1);
    }
    bind();
}

mat::mat(void) : row(0), clm(0), rc(0), dd(nullptr) { resize(0, 0); }

mat::mat(int row0, int clm0) : dd(nullptr) {
    resize(row0, clm0);
    E.setZero();
}

mat::mat(int row0, int clm0, double f) : dd(nullptr) {
    resize(row0, clm0);
    E.setConstant(f);
}

mat::mat(int row0, int clm0, double f, double f1, ...) : dd(nullptr) {
    resize(row0, clm0);
    E.setZero();
    double cur = f;
    va_list vl;
    va_start(vl, f1);
    for (int i = 0; i < rc; i++) {
        if (isInfSentinel(cur)) {
            break;
        }
        dd[i] = cur;
        if (i + 1 < rc) {
            cur = (i == 0) ? f1 : va_arg(vl, double);
        }
    }
    va_end(vl);
}

mat::mat(int row0, int clm0, const double *pf) : dd(nullptr) {
    resize(row0, clm0);
    std::memcpy(dd, pf, rc * sizeof(double));
}

mat::mat(const Eigen::MatrixXd &m) : dd(nullptr) {
    resize(static_cast<int>(m.rows()), static_cast<int>(m.cols()));
    e() = m;
}

mat::mat(const mat &o) : row(o.row), clm(o.clm), rc(o.rc), E(o.E), dd(nullptr) { bind(); }

mat &mat::operator=(const mat &o) {
    if (this == &o) {
        return *this;
    }
    row = o.row;
    clm = o.clm;
    rc  = o.rc;
    E   = o.E;
    if (rc == 0 && E.size() == 0) {
        E.resize(1, 1);
    }
    bind();
    return *this;
}

void mat::Clear(void) { E.setZero(); }

void mat::SetDiag(double f[], int len) {
    E.setZero();
    for (int i = 0, k = 0; i < row && i < clm && k < len; i++, k++) {
        if (isInfSentinel(f[k])) {
            break;
        }
        dd[i * clm + i] = f[k];
    }
}

void mat::SetDiag2(double f[], int len) {
    E.setZero();
    for (int i = 0, k = 0; i < row && i < clm && k < len; i++, k++) {
        if (isInfSentinel(f[k])) {
            break;
        }
        dd[i * clm + i] = f[k] * f[k];
    }
}

mat mat::operator+(const mat &m) const {
    assert(row == m.row && clm == m.clm);
    mat mtmp(row, clm);
    mtmp.e() = e() + m.e();
    return mtmp;
}

mat mat::operator-(const mat &m) const {
    assert(row == m.row && clm == m.clm);
    mat mtmp(row, clm);
    mtmp.e() = e() - m.e();
    return mtmp;
}

mat mat::operator*(double f) const {
    mat mtmp(row, clm);
    mtmp.e() = e() * f;
    return mtmp;
}

vect mat::operator*(const vect &v) const {
    assert(clm == v.row);
    vect vtmp(row);
    vtmp.e() = e() * v.e();
    return vtmp;
}

mat mat::operator*(const mat &m) const {
    assert(clm == m.row);
    mat mtmp(row, m.clm);
    mtmp.e() = e() * m.e();
    return mtmp;
}

mat &mat::operator=(double f) {
    E.setConstant(f);
    return *this;
}

mat &mat::operator+=(const mat &m) {
    assert(row == m.row && clm == m.clm);
    e() += m.e();
    return *this;
}

/* 把向量加到对角线上（与 ipos3g 一致） ---------------------------------------------*/
mat &mat::operator+=(const vect &v) {
    assert(row == v.row || clm == v.clm);
    int n = std::min(std::min(row, clm), v.rc);
    for (int k = 0; k < n; k++) {
        dd[k * clm + k] += v.dd[k];
    }
    return *this;
}

mat &mat::operator-=(const mat &m) {
    assert(row == m.row && clm == m.clm);
    e() -= m.e();
    return *this;
}

mat &mat::operator*=(double f) {
    e() *= f;
    return *this;
}

/* 对角线 +1 ----------------------------------------------------------------------*/
mat &mat::operator++() {
    for (int i = 0; i < row && i < clm; i++) {
        dd[i * clm + i] += 1.0;
    }
    return *this;
}

void mat::ZeroRow(int i) {
    double *p = &dd[i * clm];
    for (int j = 0; j < clm; j++) {
        p[j] = 0.0;
    }
}

void mat::ZeroClm(int j) {
    double *p = &dd[j];
    for (int i = 0; i < row; i++, p += clm) {
        *p = 0.0;
    }
}

void mat::SetRow(int i, double f, ...) {
    va_list vl;
    va_start(vl, f);
    double *p = &dd[i * clm];
    double cur = f;
    for (int j = 0; j < clm; j++) {
        p[j] = cur;
        if (j + 1 < clm) {
            cur = va_arg(vl, double);
        }
    }
    va_end(vl);
}

void mat::SetRow(int i, const vect &v) {
    assert(clm == v.clm);
    std::memcpy(&dd[i * clm], v.dd, static_cast<size_t>(clm) * sizeof(double));
}

void mat::SetClm(int j, double f[], int len) {
    double *p = &dd[j];
    for (int i = 0; i < row && i < len; i++, p += clm) {
        *p = f[i];
    }
}

void mat::SetClm(int j, const vect &v) {
    assert(row == v.row);
    double *p = &dd[j];
    for (int i = 0; i < row; i++, p += clm) {
        *p = v.dd[i];
    }
}

vect mat::GetRow(int i) const {
    vect v(1, clm);
    std::memcpy(v.dd, &dd[i * clm], static_cast<size_t>(clm) * sizeof(double));
    return v;
}

void mat::GetRow(vect &v, int i) {
    v = vect(1, clm);
    std::memcpy(v.dd, &dd[i * clm], static_cast<size_t>(clm) * sizeof(double));
}

vect mat::GetClm(int j) const {
    vect v(row, 1);
    const double *p1 = &dd[j];
    for (int i = 0; i < row; i++, p1 += clm) {
        v.dd[i] = *p1;
    }
    return v;
}

void mat::GetClm(vect &v, int j) {
    v = vect(row, 1);
    const double *p1 = &dd[j];
    for (int i = 0; i < row; i++, p1 += clm) {
        v.dd[i] = *p1;
    }
}

void mat::SetRowVect3(int i, int j, const vect3 &v) {
    double *p = &dd[i * clm + j];
    p[0]      = v.i;
    p[1]      = v.j;
    p[2]      = v.k;
}

void mat::SetRowVect3(int i, int j, const vect3 &v, const vect3 &v1) {
    double *p = &dd[i * clm + j];
    p[0]      = v.i;
    p[1]      = v.j;
    p[2]      = v.k;
    p[3]      = v1.i;
    p[4]      = v1.j;
    p[5]      = v1.k;
}

void mat::SetRowVect3(int i, int j, const vect3 &v, const vect3 &v1, const vect3 &v2) {
    double *p = &dd[i * clm + j];
    p[0]      = v.i;
    p[1]      = v.j;
    p[2]      = v.k;
    p[3]      = v1.i;
    p[4]      = v1.j;
    p[5]      = v1.k;
    p[6]      = v2.i;
    p[7]      = v2.j;
    p[8]      = v2.k;
}

void mat::SetClmVect3(int i, int j, const vect3 &v) {
    double *p = &dd[i * clm + j];
    p[0]      = v.i;
    p[clm]    = v.j;
    p[2 * clm] = v.k;
}

void mat::SetClmVect3(int i, int j, const vect3 &v, const vect3 &v1) {
    double *p = &dd[i * clm + j];
    p[0] = v.i;  p[1] = v1.i;
    p += clm;
    p[0] = v.j;  p[1] = v1.j;
    p += clm;
    p[0] = v.k;  p[1] = v1.k;
}

void mat::SetClmVect3(int i, int j, const vect3 &v, const vect3 &v1, const vect3 &v2) {
    double *p = &dd[i * clm + j];
    p[0] = v.i;  p[1] = v1.i;  p[2] = v2.i;
    p += clm;
    p[0] = v.j;  p[1] = v1.j;  p[2] = v2.j;
    p += clm;
    p[0] = v.k;  p[1] = v1.k;  p[2] = v2.k;
}

vect3 mat::GetRowVect3(int i, int j) const {
    const double *p = &dd[i * clm + j];
    return vect3(p[0], p[1], p[2]);
}

vect3 mat::GetClmVect3(int i, int j) const {
    const double *p = &dd[i * clm + j];
    return vect3(p[0], p[clm], p[2 * clm]);
}

void mat::SetDiagVect3(int i, int j, const vect3 &v) {
    double *p = &dd[i * clm + j];
    *p        = v.i;
    p += clm + 1;
    *p = v.j;
    p += clm + 1;
    *p = v.k;
}

vect3 mat::GetDiagVect3(int i, int j) const {
    if (j == -1) {
        j = i;
    }
    const double *p = &dd[i * clm + j];
    return vect3(p[0], p[clm + 1], p[2 * (clm + 1)]);
}

void mat::SetAskew(int i, int j, const vect3 &v) {
    double *p = &dd[i * clm + j];
    p[0] = 0.0;  p[1] = -v.k;  p[2] = v.j;
    p += clm;
    p[0] = v.k;  p[1] = 0.0;   p[2] = -v.i;
    p += clm;
    p[0] = -v.j; p[1] = v.i;   p[2] = 0.0;
}

void mat::SetMat3(int i, int j, const mat3 &m) {
    SetRowVect3(i, j, m.GetRow(0));
    SetRowVect3(i + 1, j, m.GetRow(1));
    SetRowVect3(i + 2, j, m.GetRow(2));
}

void mat::SetMat3(int i, int j, const mat3 &m, const mat3 &m1) {
    double *p = &dd[i * clm + j];
    for (int k = 0; k < 3; k++) {
        vect3 r0 = m.GetRow(k), r1 = m1.GetRow(k);
        p[0] = r0.i;  p[1] = r0.j;  p[2] = r0.k;
        p[3] = r1.i;  p[4] = r1.j;  p[5] = r1.k;
        p += clm;
    }
}

void mat::SetMat3(int i, int j, const mat3 &m, const mat3 &m1, const mat3 &m2) {
    double *p = &dd[i * clm + j];
    for (int k = 0; k < 3; k++) {
        vect3 r0 = m.GetRow(k), r1 = m1.GetRow(k), r2 = m2.GetRow(k);
        p[0] = r0.i;  p[1] = r0.j;  p[2] = r0.k;
        p[3] = r1.i;  p[4] = r1.j;  p[5] = r1.k;
        p[6] = r2.i;  p[7] = r2.j;  p[8] = r2.k;
        p += clm;
    }
}

mat3 mat::GetMat3(int i, int j) const {
    if (j == -1) {
        j = i;
    }
    double t[9];
    const double *p = &dd[i * clm + j];
    for (int k = 0; k < 3; k++) {
        for (int n = 0; n < 3; n++) {
            t[3 * k + n] = p[n];
        }
        p += clm;
    }
    return mat3(t);
}

void mat::SubAddMat3(int i, int j, const mat3 &m) {
    double *p = &dd[i * clm + j];
    for (int k = 0; k < 3; k++) {
        vect3 r = m.GetRow(k);
        p[0] += r.i;
        p[1] += r.j;
        p[2] += r.k;
        p += clm;
    }
}

mat operator~(const mat &m) {
    mat mtmp(m.clm, m.row);
    mtmp.e() = m.e().transpose();
    return mtmp;
}

/* vect3 的 "行向量 × 动态矩阵" 版本（ipos3g 中声明但未实现，这里补全） ---------------*/
vect3 vect3::operator*(const mat &m) const {
    assert(m.row == 3);
    Eigen::RowVectorXd r = toEigen().transpose() * m.e();
    return vect3(r(0), r(1), r(2));
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
 * vect3 工具
 * ==========================================================================*/

int IsZeros(const vect3 &v, double eps) {
    return (v.i < eps && v.i > -eps && v.j < eps && v.j > -eps && v.k < eps && v.k > -eps);
}

int IsZero(const double &val, double eps) { return (val < eps && val > -eps); }

uint8_t IsZerosXY(const vect3 &v, double eps) {
    return (v.i < eps && v.i > -eps && v.j < eps && v.j > -eps);
}

uint8_t IsNaN(const vect3 &v) {
    return (std::isnan(v.i) || std::isnan(v.j) || std::isnan(v.k)) ? 1 : 0;
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

vect3 abs(const vect3 &v) { return vect3(v.toEigen().cwiseAbs()); }

vect3 maxabs(const vect3 &v1, const vect3 &v2) {
    return vect3(v1.toEigen().cwiseAbs().cwiseMax(v2.toEigen().cwiseAbs()));
}

double norm(const vect3 &v) { return v.toEigen().norm(); }

double normInf(const vect3 &v) { return v.toEigen().lpNorm<Eigen::Infinity>(); }

double normXY(const vect3 &v) { return std::sqrt(v.i * v.i + v.j * v.j); }

double normXYInf(const vect3 &v) {
    double i = v.i > 0 ? v.i : -v.i, j = v.j > 0 ? v.j : -v.j;
    return i > j ? i : j;
}

vect3 sqrt(const vect3 &v) { return vect3(v.toEigen().cwiseSqrt()); }

vect3 pow(const vect3 &v, int k) {
    vect3 pp = v;
    for (int i = 1; i < k; i++) {
        pp.i *= v.i, pp.j *= v.j, pp.k *= v.k;
    }
    return pp;
}

double dot(const vect3 &v1, const vect3 &v2) { return v1.toEigen().dot(v2.toEigen()); }

vect3 dotmul(const vect3 &v1, const vect3 &v2) {
    return vect3(v1.toEigen().cwiseProduct(v2.toEigen()));
}

mat3 vxv(const vect3 &v1, const vect3 &v2) {
    return mat3(v1.toEigen() * v2.toEigen().transpose());
}

double sinAng(const vect3 &v1, const vect3 &v2) {
    if (IsZeros(v1) || IsZeros(v2)) {
        return 0.0;
    }
    return norm(v1 * v2) / (norm(v1) * norm(v2));
}

vect3 sort(const vect3 &v) {
    vect3 s = v;
    if (s.i < s.j) swapt(s.i, s.j);
    if (s.i < s.k) swapt(s.i, s.k);
    if (s.j < s.k) swapt(s.j, s.k);
    return s;
}

std::mt19937 &rngEngine(void) {
    static std::mt19937 engine(20240918u); /*< 固定种子，保证结果可复现 */
    return engine;
}

vect3 randn(const vect3 &mu, const vect3 &sigma) {
    std::normal_distribution<double> dist(0.0, 1.0);
    return vect3(mu.i + sigma.i * dist(rngEngine()), mu.j + sigma.j * dist(rngEngine()),
                 mu.k + sigma.k * dist(rngEngine()));
}

double MKQt(double sR, double tau) { return sR * sR * 2.0 / tau; }

vect3 MKQt(const vect3 &sR, const vect3 &tau) {
    return vect3(sR.i * sR.i * 2.0 / tau.i, sR.j * sR.j * 2.0 / tau.j, sR.k * sR.k * 2.0 / tau.k);
}

/* ============================================================================
 * 姿态 / 旋转
 * ==========================================================================*/

/* 旋转矢量 -> 四元数（小角度用泰勒展开，大角度用 Eigen::AngleAxis） ---------------*/
quat rv2q(const vect3 &rv) {
    double n2 = rv.i * rv.i + rv.j * rv.j + rv.k * rv.k;
    if (n2 < (DEG * DEG)) {
        double n4 = n2 * n2;
        double c  = 1.0 - n2 * (1.0 / 8.0) + n4 * (1.0 / 384.0);   /*< cos(|rv|/2) */
        double f  = 0.5 - n2 * (1.0 / 48.0) + n4 * (1.0 / 3840.0); /*< sin(|rv|/2)/|rv| */
        return quat(c, f * rv.i, f * rv.j, f * rv.k);
    }
    double th = std::sqrt(n2);
    return quat(Eigen::Quaterniond(Eigen::AngleAxisd(th, rv.toEigen() / th)));
}

mat3 rv2m(const vect3 &rv) { return q2mat(rv2q(rv)); }

mat3 q2mat(const quat &qnb) { return mat3(qnb.toEigen().toRotationMatrix()); }

quat m2qua(const mat3 &Cnb) {
    Eigen::Quaterniond q(Cnb.e());
    q.normalize();
    /* 与 ipos3g 一致：固定 q0 >= 0 的分支 */
    if (q.w() < 0) {
        q.coeffs() = -q.coeffs();
    }
    return quat(q);
}

mat3 askew(const vect3 &v) {
    /*
     *      [  0  -k   j ]
     *      [  k   0  -i ]
     *      [ -j   i   0 ]
     */
    return mat3(0, -v.k, v.j, v.k, 0.0, -v.i, -v.j, v.i, 0);
}

/* 欧拉角 [pitch,roll,yaw] -> Cnb（ipos3g / PSINS 约定，保留闭式） -----------------*/
mat3 a2mat(const vect3 &att) {
    double si = std::sin(att.i), ci = std::cos(att.i), sj = std::sin(att.j), cj = std::cos(att.j),
           sk = std::sin(att.k), ck = std::cos(att.k);
    return mat3(cj * ck - si * sj * sk, -ci * sk, sj * ck + si * cj * sk,
                cj * sk + si * sj * ck, ci * ck, sj * sk - si * cj * ck,
                -ci * sj, si, ci * cj);
}

vect3 m2att(const mat3 &Cnb) {
    return vect3(asinEx(Cnb.e21), atan2Ex(-Cnb.e20, Cnb.e22), atan2Ex(-Cnb.e01, Cnb.e11));
}

/* 反序欧拉角 -> Cnb（保留闭式） ---------------------------------------------------*/
mat3 ar2mat(const vect3 &attr) {
    double si = std::sin(attr.i), ci = std::cos(attr.i), sj = std::sin(attr.j), cj = std::cos(attr.j),
           sk = std::sin(attr.k), ck = std::cos(attr.k);
    return mat3(cj * ck, si * sj * ck - ci * sk, ci * sj * ck + si * sk,
                cj * sk, si * sj * sk + ci * ck, ci * sj * sk - si * ck,
                -sj, si * cj, ci * cj);
}

quat ar2qua(const vect3 &attr) { return m2qua(ar2mat(attr)); }

vect3 m2attr(const mat3 &Cnb) {
    return vect3(atan2Ex(Cnb.e21, Cnb.e22), asinEx(-Cnb.e20), atan2Ex(Cnb.e10, Cnb.e00));
}

vect3 q2attr(const quat &qnb) { return m2attr(q2mat(qnb)); }

/* 欧拉角 [pitch,roll,yaw] -> 四元数 -----------------*/
quat a2qua(double pitch, double roll, double yaw) {
    pitch /= 2.0, roll /= 2.0, yaw /= 2.0;
    double sp = std::sin(pitch), sr = std::sin(roll), sy = std::sin(yaw),
           cp = std::cos(pitch), cr = std::cos(roll), cy = std::cos(yaw);
    return quat(cp * cr * cy - sp * sr * sy,
                sp * cr * cy - cp * sr * sy,
                cp * sr * cy + sp * cr * sy,
                cp * cr * sy + sp * sr * cy);
}

quat a2qua(const vect3 &att) { return a2qua(att.i, att.j, att.k); }

vect3 q2att(const quat &qnb) {
    double q11 = qnb.q0 * qnb.q0, q12 = qnb.q0 * qnb.q1, q13 = qnb.q0 * qnb.q2,
           q14 = qnb.q0 * qnb.q3, q22 = qnb.q1 * qnb.q1, q23 = qnb.q1 * qnb.q2,
           q24 = qnb.q1 * qnb.q3, q33 = qnb.q2 * qnb.q2, q34 = qnb.q2 * qnb.q3,
           q44 = qnb.q3 * qnb.q3;
    return vect3(asinEx(2 * (q34 + q12)), atan2Ex(-2 * (q24 - q13), q11 - q22 - q33 + q44),
                 atan2Ex(-2 * (q23 - q14), q11 - q22 + q33 - q44));
}
#if 0
/* 四元数 -> 旋转矢量 --------------------------------------------------------------*/
vect3 q2rv(const quat &q) {
    quat dq = q;
    if (dq.q0 < 0) {
        dq.q0 = -dq.q0, dq.q1 = -dq.q1, dq.q2 = -dq.q2, dq.q3 = -dq.q3;
    }
    if (dq.q0 > 1.0) {
        dq.q0 = 1.0;
    }
    double n2 = std::acos(dq.q0), f;
    if (n2 > 1.0e-20) {
        f = 2.0 / (std::sin(n2) / n2);
    } else {
        f = 2.0;
    }
    return vect3(dq.q1, dq.q2, dq.q3) * f;
}
#endif
vect3 q2rv(const quat &q) {
    quat dq = q;
    double n = std::sqrt(dq.q0*dq.q0 + dq.q1*dq.q1 + dq.q2*dq.q2 + dq.q3*dq.q3);
    if (n > 0.0) { dq.q0 /= n; dq.q1 /= n; dq.q2 /= n; dq.q3 /= n; }
    /* 规范化：让 q0 >= 0；若 q0 == 0，让首个非零虚部 >= 0 */
    if (dq.q0 < 0.0 ||  (dq.q0 == 0.0 &&(dq.q1 < 0.0 || (dq.q1 == 0.0 && (dq.q2 < 0.0 || (dq.q2 == 0.0 && dq.q3 < 0.0)))))) {
        dq.q0 = -dq.q0; dq.q1 = -dq.q1; dq.q2 = -dq.q2; dq.q3 = -dq.q3;
    }
    double s = std::sqrt(std::max(0.0, 1.0 - dq.q0 * dq.q0));
    double theta = 2.0 * std::acos(range(dq.q0, -1.0, 1.0));
    double k = (theta < 1.0e-20) ? 2.0 : theta / s;
    return vect3(dq.q1 * k, dq.q2 * k, dq.q3 * k);
}

/* 旋转矩阵 -> 旋转矢量 ------------------------------------------------------------*/
vect3 m2rv(const mat3 &Cnb) {
    double phi = std::acos(range((trace(Cnb) - 1.0) / 2.0, -1.0, 1.0));
    double afa = (-1.0e-10 < phi && phi < 1.0e-10) ? 0.5 : phi / (2.0 * std::sin(phi));
    return vect3(Cnb.e21 - Cnb.e12, Cnb.e02 - Cnb.e20, Cnb.e10 - Cnb.e01) * afa;
}

vect3 qq2phi(const quat &qcalcu, const quat &qreal) { return q2rv(qreal * (~qcalcu)); }

quat addmu(const quat &q, const vect3 &mu) { return q * rv2q(mu); }

quat UpDown(const quat &q) {
    vect3 att = q2att(q);
    att.i     = -att.i;
    att.j += PI;
    return a2qua(att);
}

vect3 sv2att(const vect3 &fb, double yaw0, const vect3 &fn) {
    vect3 phi = fb * fn;
    double afa = std::acos(range(dot(fn, fb) / norm(fn) / norm(fb), -1.0, 1.0)), nphi = norm(phi);
    vect3 att = q2att(rv2q(phi * (nphi < 1.0e-10 ? 1.0 : afa / nphi)));
    att.k     = yaw0;
    return att;
}

vect3 vn2att(const vect3 &vn) {
    double vel = normXY(vn);
    if (vel < 1.0e-6) {
        return O31;
    }
    return vect3(std::atan2(vn.k, vel), 0, std::atan2(-vn.i, vn.j));
}

double vn2att(double vel_east, double vel_north) {
    double vel = vel_east * vel_east + vel_north * vel_north;
    if (vel < 3) {
        return 0;
    }
    return std::atan2(-vel_east, vel_north);
}

/* ============================================================================
 * mat3 线性代数
 * ==========================================================================*/

mat3 Rot(double angle, char axis) {
    double s = std::sin(angle), c = std::cos(angle);
    switch (axis) {
    case 'x':
    case 'X': return mat3(1, 0, 0, 0, c, -s, 0, s, c);
    case 'y':
    case 'Y': return mat3(c, 0, s, 0, 1, 0, -s, 0, c);
    default: return mat3(c, -s, 0, s, c, 0, 0, 0, 1);
    }
}

mat3 rcijk(const mat3 &m, int ijk) {
    switch (ijk) {
    case 021: return mat3(m.e00, m.e02, m.e01, m.e20, m.e22, m.e21, m.e10, m.e12, m.e11);
    case 102: return mat3(m.e11, m.e10, m.e12, m.e01, m.e00, m.e02, m.e21, m.e20, m.e22);
    case 120: return mat3(m.e11, m.e12, m.e10, m.e21, m.e22, m.e20, m.e01, m.e02, m.e00);
    case 201: return mat3(m.e22, m.e20, m.e21, m.e02, m.e00, m.e01, m.e12, m.e10, m.e11);
    case 210: return mat3(m.e22, m.e21, m.e20, m.e12, m.e11, m.e10, m.e02, m.e01, m.e00);
    default: return m;
    }
}

double trMMT(const mat3 &m1, const mat3 &m2) {
    const mat3 &b = (&m2 == &I33) ? m1 : m2;
    return (m1.e().cwiseProduct(b.e())).sum();
}

void symmetry(mat3 &m) {
    m.e01 = m.e10 = (m.e01 + m.e10) * 0.5;
    m.e02 = m.e20 = (m.e02 + m.e20) * 0.5;
    m.e12 = m.e21 = (m.e12 + m.e21) * 0.5;
}

mat3 pow(const mat3 &m, int k) {
    mat3 mm = m;
    for (int i = 1; i < k; i++) {
        mm = mm * m;
    }
    return mm;
}

double trace(const mat3 &m) { return m.e().trace(); }

double det(const mat3 &m) { return m.e().determinant(); }

mat3 adj(const mat3 &m) {
    /* 不用 Eigen 的 adjugate()：
     *   1) 它靠 EIGEN_MATRIXBASE_PLUGIN 注入到 MatrixBase，
     *      对 Map<> 类型的可见性在不同 3.4.x 版本/包含顺序下不一致；
     *   2) 3x3 伴随矩阵直接展开也就 9 个乘减，没必要依赖库实现。
     * 伴随矩阵定义：A·adj(A) = det(A)·I，元素为代数余子式的转置。 */
    const Eigen::Matrix3d A = m.e();     /* Map -> 具体矩阵（列主序也可） */
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
    return mat3(a);
}

mat3 inv(const mat3 &m) { return mat3(m.e().inverse()); }

vect3 diag(const mat3 &m) { return vect3(m.e00, m.e11, m.e22); }

mat3 diag(const vect3 &v) { return mat3(v.i, 0, 0, 0, v.j, 0, 0, 0, v.k); }

mat3 askew(const mat3 &m, int I) {
    mat3 m1;
    m1.e01 = (m.e01 - m.e10) / 2;
    m1.e02 = (m.e02 - m.e20) / 2;
    m1.e12 = (m.e12 - m.e21) / 2;
    m1.e10 = -m1.e01;
    m1.e20 = -m1.e02;
    m1.e21 = -m1.e12;
    if (I == 0) {
        m1.e00 = m1.e11 = m1.e22 = 0.0;
    } else if (I == 1) {
        m1.e00 = m1.e11 = m1.e22 = 1.0;
    } else {
        m1.e00 = m.e00, m1.e11 = m.e11, m1.e22 = m.e22;
    }
    return m1;
}

mat3 dotmul(const mat3 &m1, const mat3 &m2) { return mat3(m1.e().cwiseProduct(m2.e())); }

mat3 MMT(const mat3 &m1, const mat3 &m2) {
    const mat3 &b = (&m2 == &I33) ? m1 : m2;
    return mat3(m1.e() * b.e().transpose());
}

double norm(const mat3 &m) { return std::sqrt(trMMT(m)); }

mat3 randn(const mat3 &mu, const double &sigma) {
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::Matrix3d r = mu.toEigen();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            r(i, j) += sigma * dist(rngEngine());
        }
    }
    return mat3(r);
}

/* ============================================================================
 * vect 线性代数
 * ==========================================================================*/

vect abs(const vect &v) {
    vect res(v.row, v.clm);
    res.e() = v.e().cwiseAbs();
    return res;
}

double norm(const vect &v) { return v.toEigen().norm(); }

double norm1(const vect &v) { return v.toEigen().lpNorm<1>(); }

double normInf(const vect &v) { return v.toEigen().lpNorm<Eigen::Infinity>(); }

vect pow(const vect &v, int k) {
    vect pp = v;
    for (int i = 1; i < k; i++) {
        pp.e() = pp.e().cwiseProduct(v.e());
    }
    return pp;
}

vect sort(const vect &v) {
    vect vtmp = v;
    std::sort(vtmp.dd, vtmp.dd + vtmp.rc, std::greater<double>());
    return vtmp;
}

double dot(const vect &v1, const vect &v2) {
    assert(v1.row == v2.row && v1.clm == v2.clm);
    return v1.toEigen().dot(v2.toEigen());
}

vect dotmul(const vect &v1, const vect &v2) {
    assert(v1.row == v2.row && v1.clm == v2.clm);
    vect res(v1.row, v1.clm);
    res.e() = v1.e().cwiseProduct(v2.e());
    return res;
}

vect randn(const vect &mu, const vect &sigma) {
    assert(mu.rc == sigma.rc);
    vect res(mu.row, mu.clm);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < res.rc; i++) {
        res.dd[i] = mu.dd[i] + sigma.dd[i] * dist(rngEngine());
    }
    return res;
}

/* ============================================================================
 * mat 线性代数
 * ==========================================================================*/

void symmetry(mat &m) {
    assert(m.row == m.clm);
    for (int i = 0; i < m.row; i++) {
        for (int j = i + 1; j < m.clm; j++) {
            double f = (m.dd[i * m.clm + j] + m.dd[j * m.clm + i]) * 0.5;
            m.dd[i * m.clm + j] = f;
            m.dd[j * m.clm + i] = f;
        }
    }
}

double trace(const mat &m) {
    assert(m.row == m.clm);
    double s = 0.0;
    for (int i = 0; i < m.row; i++) {
        s += m.dd[i * m.clm + i];
    }
    return s;
}

double norm1(const mat &m) { return norm1(&m.dd[0], m.rc); }

double normInf(const mat &m) { return normInf(&m.dd[0], m.rc); }

mat dotmul(const mat &m1, const mat &m2) {
    assert(m1.row == m2.row && m1.clm == m2.clm);
    mat res(m1.row, m1.clm);
    res.e() = m1.e().cwiseProduct(m2.e());
    return res;
}

vect diag(const mat &m) {
    vect vtmp(m.row, 1);
    /* 行主序下第 i 个对角元在 dd[i*(clm+1)]。真正的对角元只有 min(row,clm) 个，
     * 对"高瘦"矩阵（row > clm）直接按 clm+1 步长会越界读，因此超出部分补 0。 */
    const int n = (m.row < m.clm) ? m.row : m.clm;
    for (int i = 0; i < m.row; i++) {
        vtmp.dd[i] = (i < n) ? m.dd[i * (m.clm + 1)] : 0.0;
    }
    return vtmp;
}

mat diag(const vect &v) {
    int rc = v.row > v.clm ? v.row : v.clm;
    mat mtmp(rc, rc, 0.0);
    for (int i = 0; i < rc; i++) {
        mtmp.dd[i * rc + i] = v.dd[i];
    }
    return mtmp;
}

mat eye(int n) {
    mat m(n, n, 0.0);
    for (int i = 0; i < n; i++) {
        m.dd[i * n + i] = 1.0;
    }
    return m;
}

mat inv4(const mat &m) {
    assert(m.clm == m.row && m.clm == 4);
    mat r(4, 4);
    r.e() = m.e().inverse();
    return r;
}

void RowMul(mat &m, const mat &m0, const mat &m1, int r, int fast) {
    assert(m0.clm == m1.row);
    int rc0 = r * m0.clm;
    fast    = (r >= fast);
    double *p = &m.dd[rc0];
    for (int j = 0; j < m0.clm; j++) {
        if (fast) {
            p[j] = m1.dd[r * m1.clm + j];
            continue;
        }
        double f = 0.0;
        for (int k = 0; k < m0.clm; k++) {
            f += m0.dd[rc0 + k] * m1.dd[k * m1.clm + j];
        }
        p[j] = f;
    }
}

void RowMulT(mat &m, const mat &m0, const mat &m1, int r, int fast) {
    assert(m0.clm == m1.clm);
    int rc0  = r * m0.clm;
    double *p = &m.dd[rc0];
    const double *p0 = &m0.dd[rc0];
    const double *p1jk = m1.dd;
    for (int j = 0; j < m0.clm; j++) {
        if (j >= fast) {
            p[j] = p0[j];
            p1jk += m1.clm;
            continue;
        }
        double f = 0.0;
        for (int k = 0; k < m0.clm; k++, p1jk++) {
            f += p0[k] * (*p1jk);
        }
        p[j] = f;
    }
}

void DVMDVafa(const vect &V, mat &M, double afa) {
    assert(V.rc == M.row && M.row == M.clm);
    for (int i = 0; i < M.clm; i++) {
        double vi = V.dd[i], viafa = vi * afa;
        double *prow = &M.dd[i * M.clm];
        for (int j = 0; j < M.clm; j++) {
            prow[j] *= vi;
        }
        double *pclm = &M.dd[i];
        for (int j = 0; j < M.clm; j++, pclm += M.row) {
            *pclm *= viafa;
        }
    }
}

mat randn(const mat &mu, const double &sigma) {
    mat res(mu.row, mu.clm);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < mu.rc; i++) {
        res.dd[i] = mu.dd[i] + sigma * dist(rngEngine());
    }
    return res;
}

/* ============================================================================
 * 坐标系
 * ==========================================================================*/

mat3 pos2Cen(const vect3 &pos) {
    double si = std::sin(pos.i), ci = std::cos(pos.i), sj = std::sin(pos.j), cj = std::cos(pos.j);
    return mat3(-sj, -si * cj, ci * cj, cj, -si * sj, ci * sj, 0, ci, si);
}

vect3 xyz2blh(const vect3 &xyz) {
    double s     = normXY(xyz);
    /*
     * Bowring/Heiskanen 参数纬度：theta = atan2(z*a, b*p)，p = sqrt(x^2+y^2) = s
     */
    double theta = std::atan2(xyz.k * RE, s * RP);
    double s3    = std::sin(theta), c3 = std::cos(theta);
    s3 = s3 * s3 * s3;
    c3 = c3 * c3 * c3;
    if (s < 1.0) { return O31; }
    double L  = std::atan2(xyz.j, xyz.i);
    double B  = std::atan2(xyz.k + (ep2 * RP * s3), s - e2 * RE * c3);
    double sB = std::sin(B), cB = std::cos(B);
    double N  = RE / std::sqrt(1 - e2 * sB * sB);
    return vect3(B, L, s / cB - N);
}

vect3 blh2xyz(const vect3 &blh) {
    double sB = std::sin(blh.i), cB = std::cos(blh.i), sL = std::sin(blh.j), cL = std::cos(blh.j),
           N = RE / std::sqrt(1 - e2 * sB * sB);
    return vect3((N + blh.k) * cB * cL, (N + blh.k) * cB * sL, (N * (1 - e2) + blh.k) * sB);
}

vect3 Vxyz2enu(const vect3 &Vxyz, const vect3 &pos) { return Vxyz * pos2Cen(pos); }

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
    vect3 att1;

    /* n < 100 时前 3 个量按"姿态"处理（与 ipos3g 一致，含对 x2 的写回） */
    if (n < 100) {
        x10  = x1;
        xf0  = xf;
        att1 = vect3(x1[0], x1[1], x1[2]);
        vect3 phi = qq2phi(a2qua(vect3(x2[0], x2[1], x2[2])), a2qua(att1));
        double *x2w = const_cast<double *>(x2);
        x2w[0] = phi.i;
        x2w[1] = phi.j;
        x2w[2] = phi.k;
        x1[0] = x1[1] = x1[2] = 0.0;
    }

    int j = (n > 100) ? 100 : 0;
    for (; j < n; j++, x1++, p1++, x2++, p2++, xf++, pf++) {
        double p1p2 = *p1 + *p2;
        *xf = (*p1 * (*x2) + *p2 * (*x1)) / p1p2;
        *pf = *p1 * *p2 / p1p2;
    }

    if (n < 100 && xf0 != nullptr) {
        vect3 v = q2att(a2qua(att1) + vect3(xf0[0], xf0[1], xf0[2]));
        xf0[0] = v.i;
        xf0[1] = v.j;
        xf0[2] = v.k;
        if (xf0 != x10) {
            x10[0] = att1.i;
            x10[1] = att1.j;
            x10[2] = att1.k;
        }
    }
}

void fusion(vect3 &x1, vect3 &p1, const vect3 x2, const vect3 p2) {
    fusion(&x1.i, &p1.i, &x2.i, &p2.i, 3);
}

void fusion(vect3 &x1, vect3 &p1, const vect3 x2, const vect3 p2, vect3 &xf, vect3 &pf) {
    fusion(&x1.i, &p1.i, &x2.i, &p2.i, 3, &xf.i, &pf.i);
}

/* ============================================================================
 * 扩展：对称正定线性代数（Eigen）
 * ==========================================================================*/

mat llt_L(const mat &A) {
    if (A.row != A.clm || A.row == 0) {
        return mat(0, 0);
    }
    /* 注意必须 .eval()：Eigen 对 `a = a + a.transpose()` 这种自别名写法
     * 不会自动插入临时变量，逐个元素赋值会读到已被覆盖的转置项。 */
    Eigen::MatrixXd a = (0.5 * (A.toEigen() + A.toEigen().transpose())).eval();

    /* 快路径：Cholesky */
    Eigen::LLT<Eigen::MatrixXd> llt(a);
    if (llt.info() == Eigen::Success) {
        return mat(Eigen::MatrixXd(llt.matrixL()));
    }

    /* 慢路径：预积分的 Qk 用的是梯形离散
     *     Qk = 0.5*dt*(phi·A + A·phiᵀ)
     * 它在 (p,v) 分块上会带来一个极小的负特征值（量级 ~1e-5·trace），
     * 于是 covariance_ 在积分步数很少时会轻微不正定，LLT 直接失败。
     * 这里改用 LDLT，并把 D 里低于"相对阈值"的元素钳到阈值上：
     *     A = Pᵀ L D Lᵀ P   ->   L~ = Pᵀ · L · sqrt(max(D, thr))
     * 得到 L~ L~ᵀ ≈ A（负方向被投影成一个小正数）。
     * 阈值取相对量 1e-12·max|D|，因此协方差整体放大/缩小时都有效；
     * 旧实现用固定 1e-300 阻尼，对这种量级的负特征值完全无效。 */
    Eigen::LDLT<Eigen::MatrixXd> ldlt(a);
    if (ldlt.info() != Eigen::Success) {
        return mat(0, 0);
    }
    const Eigen::VectorXd d = ldlt.vectorD();
    const double dmax       = d.cwiseAbs().maxCoeff();
    if (!(dmax > 0.0) || !std::isfinite(dmax)) {
        return mat(0, 0);
    }
    const double thr = 1.0e-12 * dmax;
    Eigen::VectorXd s(d.size());
    for (int i = 0; i < d.size(); i++) {
        s(i) = std::sqrt(d(i) > thr ? d(i) : thr);
    }
    const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm(ldlt.transpositionsP());
    const Eigen::MatrixXd Ld = Eigen::MatrixXd(ldlt.matrixL());
    Eigen::MatrixXd L = (perm.transpose() * Ld * s.asDiagonal()).eval();
    return mat(L);
}


mat llt_sqrtinv(const mat &A) {
    mat L = llt_L(A);
    if (L.row > 0) {
        Eigen::MatrixXd Le   = L.toEigen();
        Eigen::MatrixXd Linv = Le.triangularView<Eigen::Lower>().solve(
            Eigen::MatrixXd::Identity(Le.rows(), Le.cols()));
        /* S = L^{-1}（下三角），S^T S = L^{-T} L^{-1} = A^{-1} */
        if (Linv.allFinite()) {
            return mat(Eigen::MatrixXd(Linv));   // 原为 Linv.transpose()
        }
    }

    /* 兜底：A 严重退化时（例如刚 reset、协方差里还有数值负特征值），
     * 用特征分解求带相对截断的伪逆平方根：S = V·diag(1/sqrt(d))·Vᵀ。
     * 与 pinv_sym / schur_complement 的处理方式保持一致。 */
    if (A.row != A.clm || A.row == 0) {
        return mat(0, 0);
    }
    Eigen::MatrixXd a = (0.5 * (A.toEigen() + A.toEigen().transpose())).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(a);
    if (es.info() != Eigen::Success) {
        return mat(0, 0);
    }
    const Eigen::VectorXd d = es.eigenvalues();
    const Eigen::MatrixXd V = es.eigenvectors();
    const double dmax       = d.cwiseAbs().maxCoeff();
    if (!(dmax > 0.0) || !std::isfinite(dmax)) {
        return mat(0, 0);
    }
    const double thr = 1.0e-10 * dmax;
    Eigen::VectorXd dsinv(d.size());
    for (int i = 0; i < d.size(); i++) {
        dsinv(i) = (d(i) > thr) ? 1.0 / std::sqrt(d(i)) : 0.0;
    }
    return mat(Eigen::MatrixXd(V * dsinv.asDiagonal() * V.transpose()));
}

mat llt_inv(const mat &A) {
    mat L = llt_L(A);
    if (L.row == 0) {
        return mat(0, 0);
    }
    Eigen::MatrixXd Le = L.toEigen();
    Eigen::MatrixXd Linv = Le.triangularView<Eigen::Lower>().solve(
        Eigen::MatrixXd::Identity(Le.rows(), Le.cols()));
    return mat(Eigen::MatrixXd(Linv.transpose() * Linv));
}

mat pinv_sym(const mat &A, double eps) {
    if (A.row != A.clm || A.row == 0) {
        return mat(0, 0);
    }
    Eigen::MatrixXd a = A.toEigen();
    a                 = (0.5 * (a + a.transpose())).eval(); /*< 必须 eval，见 llt_L 的注释 */

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(a);
    if (es.info() != Eigen::Success) {
        return mat(0, 0);
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
    return mat(Eigen::MatrixXd(V * dinv.asDiagonal() * V.transpose()));
}


void LocalFrame::init(const vect3 &blh0) {
    blh0_ = blh0;
    eth_.Update(blh0_, O31);
    gravity_ = eth_.gn;
    init_    = true;
}

vect3 LocalFrame::blh2neu(const vect3 &blh) const {
    double pn = (blh.i - blh0_.i) * eth_.RMh;
    double pe = (blh.j - blh0_.j) * eth_.clRNh;
    double pu = blh.k - blh0_.k;
    return vect3(pe, pn, pu);
}

vect3 LocalFrame::neu2blh(const vect3 &neu) const {
    double lat = blh0_.i + neu.j * eth_.f_RMh;
    double lon = blh0_.j + neu.i * eth_.f_clRNh;
    double hgt = blh0_.k + neu.k;
    return vect3(lat, lon, hgt);
}

vect3 quatVec(const quat &q) { return vect3(q.q1, q.q2, q.q3); }

/* pose = {pE, pN, pU, qx, qy, qz, qw}
 * mix  = {v, bg, ba} 或 {v, bg, ba, sodo} ----------------------------------------*/
void stateToData(const State &state, double *pose, double *mix, bool with_odometer) {
    pose[0] = state.p.i;
    pose[1] = state.p.j;
    pose[2] = state.p.k;
    pose[3] = state.q.q1; /*< x */
    pose[4] = state.q.q2; /*< y */
    pose[5] = state.q.q3; /*< z */
    pose[6] = state.q.q0; /*< w */

    mix[0] = state.v.i;
    mix[1] = state.v.j;
    mix[2] = state.v.k;
    mix[3] = state.bg.i;
    mix[4] = state.bg.j;
    mix[5] = state.bg.k;
    mix[6] = state.ba.i;
    mix[7] = state.ba.j;
    mix[8] = state.ba.k;
    if (with_odometer) {
        mix[9] = state.sodo;
    }
}

void stateFromData(const double *pose, const double *mix, bool with_odometer, State &state) {
    state.p = vect3(pose[0], pose[1], pose[2]);
    state.q = quat(pose[6], pose[3], pose[4], pose[5]);
    normlize(&state.q);

    state.v  = vect3(mix[0], mix[1], mix[2]);
    state.bg = vect3(mix[3], mix[4], mix[5]);
    state.ba = vect3(mix[6], mix[7], mix[8]);
    state.sodo = with_odometer ? mix[9] : 0.0;
}

void posePlus(const double *pose, const double *delta, double *pose_plus) {
    quat q(pose[6], pose[3], pose[4], pose[5]);
    normlize(&q);

    quat q_plus = q * rv2q(vect3(delta[3], delta[4], delta[5]));
    normlize(&q_plus);

    pose_plus[0] = pose[0] + delta[0];
    pose_plus[1] = pose[1] + delta[1];
    pose_plus[2] = pose[2] + delta[2];
    pose_plus[3] = q_plus.q1;
    pose_plus[4] = q_plus.q2;
    pose_plus[5] = q_plus.q3;
    pose_plus[6] = q_plus.q0;
}

void poseMinus(const double *y, const double *x, double *y_minus_x) {
    quat qy(y[6], y[3], y[4], y[5]);
    quat qx(x[6], x[3], x[4], x[5]);
    normlize(&qy);
    normlize(&qx);

    //vect3 dphi = q2rv(qy * (~qx)); /*< qx -> qy 的旋转矢量 */
    vect3 dphi = q2rv((~qx) * qy);   /*< log(qx⁻¹ ⊗ qy)，与 posePlus 对偶 */

    y_minus_x[0] = y[0] - x[0];
    y_minus_x[1] = y[1] - x[1];
    y_minus_x[2] = y[2] - x[2];
    y_minus_x[3] = dphi.i;
    y_minus_x[4] = dphi.j;
    y_minus_x[5] = dphi.k;
}


mat quatLeftMatrix(const quat &q) {
    mat m(4, 4, 0.0);
    m.dd[0]  = q.q0;
    m.dd[1]  = -q.q1;
    m.dd[2]  = -q.q2;
    m.dd[3]  = -q.q3;
    m.dd[4]  = q.q1;
    m.dd[5]  = q.q0;
    m.dd[6]  = -q.q3;
    m.dd[7]  = q.q2;
    m.dd[8]  = q.q2;
    m.dd[9]  = q.q3;
    m.dd[10] = q.q0;
    m.dd[11] = -q.q1;
    m.dd[12] = q.q3;
    m.dd[13] = -q.q2;
    m.dd[14] = q.q1;
    m.dd[15] = q.q0;
    return m;
}

mat quatRightMatrix(const quat &q) {
    mat m(4, 4, 0.0);
    m.dd[0]  = q.q0;
    m.dd[1]  = -q.q1;
    m.dd[2]  = -q.q2;
    m.dd[3]  = -q.q3;
    m.dd[4]  = q.q1;
    m.dd[5]  = q.q0;
    m.dd[6]  = q.q3;
    m.dd[7]  = -q.q2;
    m.dd[8]  = q.q2;
    m.dd[9]  = -q.q3;
    m.dd[10] = q.q0;
    m.dd[11] = q.q1;
    m.dd[12] = q.q3;
    m.dd[13] = q.q2;
    m.dd[14] = -q.q1;
    m.dd[15] = q.q0;
    return m;
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

void IMU::SetKga(const mat3 &Kg0, const vect3 eb0, const mat3 &Ka0, const vect3 &db0) {
    Kg = Kg0;
    eb = eb0;
    Ka = Ka0;
    db = db0;
}

double IMU::Update(const vect3 *pwm, const vect3 *pvm, int nSamples0, double ts) {
    assert(nSamples0 > 0 && nSamples0 < 6);

    int i;
    const double *pcf = (nSamples0 >= 2) ? conefactors[nSamples0 - 2] : nullptr;
    vect3 cm(0.0), sm(0.0);
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

    phim = wmm + cm * pwm[i];
    dvbm = vmm + wmm * (vmm * 0.5) + (cm * pvm[i] + sm * pwm[i]);

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

void earth::Update(const vect3 &pos0, const vect3 &vn0) {
    pos = pos0;
    vn  = vn0;

    sl = std::sin(pos.i);
    cl = std::cos(pos.i);
    tl = sl / cl;

    double sq  = 1.0 - e2 * sl * sl;
    double sq2 = std::sqrt(sq);

    RMh    = a * (1.0 - e2) / sq / sq2 + pos.k;
    f_RMh  = 1.0 / RMh;
    RNh    = a / sq2 + pos.k;
    clRNh  = cl * RNh;
    f_RNh  = 1.0 / RNh;
    f_clRNh = 1.0 / clRNh;

    wnie = vect3(0.0, wie * cl, wie * sl);
    wnen = vect3(-vn.j * f_RMh, vn.i * f_RNh, vn.i * f_RNh * tl);
    wnin = wnie + wnen;

    sl2 = sl * sl;
    sl4 = sl2 * sl2;
    gn  = vect3(0.0, 0.0, -(G0 * (1.0 + 5.27094e-3 * sl2 + 2.32718e-5 * sl4) - 3.086e-6 * pos.k));

    gcc = pgn ? *pgn : gn;
    gcc -= (wnie + wnin) * vn; /*< 叉乘：-(wie+wen) x vn */
}

vect3 earth::vn2dpos(const vect3 &vn0, float ts) const {
    return vect3(vn0.j * f_RMh, vn0.i * f_clRNh, vn0.k) * static_cast<double>(ts);
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

void Preintegration::setBias(const vect3 &bg, const vect3 &ba) {
    const double dbg = (bg - bg_).toEigen().norm();
    const double dba = (ba - ba_).toEigen().norm();
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
    const vect3 bg_save = bg_;
    const vect3 ba_save = ba_;
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
    const vect3 dtheta = meas.dtheta - bg_ * dt;
    const vect3 dvel   = meas.dvel   - ba_ * dt;

    const Eigen::Vector3d dtheta_e = dtheta.toEigen();
    const Eigen::Vector3d dvel_e   = dvel.toEigen();

    /* 单步旋转 & 半拍旋转 */
    quat dq_step = rv2q(dtheta);
    dq_step.normlize(&dq_step);

    quat dq_half = rv2q(dtheta * 0.5);
    dq_half.normlize(&dq_half);

    const quat q_prev = q_;
    quat q_mid = q_prev * dq_half;
    q_mid.normlize(&q_mid);

    Eigen::Matrix3d R_prev = q_prev.toEigen().toRotationMatrix();
    Eigen::Matrix3d R_mid  = q_mid .toEigen().toRotationMatrix();

    const Eigen::Vector3d a_body = dvel_e / dt;

    /* 位置/速度更新 */
    const vect3 v_old = v_;
    const Eigen::Vector3d a_b0 = R_mid * a_body;

    p_ = p_ + v_old * dt + vect3(a_b0 * (0.5 * dt * dt));
    v_ = v_old + vect3(a_b0 * dt);

    q_ = q_prev * dq_step;
    q_.normlize(&q_);

    /* 协方差传播 */
    const Eigen::Vector3d w_body = dtheta_e / dt;
    CovMatrix   F = buildF(a_body, w_body, R_prev, dt);
    GainMatrix  G = buildG(R_prev);
    NoiseMatrix Q = buildQ(dt);

    cov_ = F * cov_ * F.transpose() + G * Q * G.transpose();
    jac_ = F * jac_;

    if (meas.odovel != 0.0) {
        /* 安装角 C_b^m * e_x。abv = [pitch, 0, yaw] */
        const double ap = param_.abv.i;   /* pitch */
        const double ay = param_.abv.j;   /* yaw   */
        const Eigen::Vector3d cx(
             std::cos(ay) * std::cos(ap),
            -std::sin(ay) * std::cos(ap),
            -std::sin(ap));

        /* 刻度因子 */
        const double ds_scaled = (1.0 + sodo_) * meas.odovel;

        /* 杆臂补偿：omega_ib^b × l_OD；dtheta_e 是已经去偏后的角增量 */
        const Eigen::Vector3d w_body_b = dtheta_e / dt;
        const Eigen::Vector3d v_lev =  w_body_b.cross(param_.lvOD.toEigen());

        const Eigen::Vector3d ds_body = ds_scaled * cx + v_lev * dt;
        s_ = s_ + vect3(R_mid * ds_body);
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

    const vect3 dp  (delta(0),  delta(1),  delta(2) );
    const vect3 dphi(delta(3),  delta(4),  delta(5) );
    const vect3 dv  (delta(6),  delta(7),  delta(8) );
    const vect3 dbg (delta(9),  delta(10), delta(11));
    const vect3 dba (delta(12), delta(13), delta(14));

    /* 平移 / 速度 / 零偏：直接相加 */
    r.p  = s.p  + dp;
    r.v  = s.v  + dv;
    r.bg = s.bg + dbg;
    r.ba = s.ba + dba;

    /* 旋转：右扰动，扰动向量在体轴系 */
    r.q  = s.q * rv2q(dphi);
    r.q.normlize(&r.q);    /* 防止长期迭代中数值漂移 */

    return r;
}

DeltaN minus(const State &a, const State &b) {
    DeltaN d;
    d.setZero();

    /* 平移 / 速度 / 零偏：直接相减 */
    d.segment<3>(0)  = (a.p  - b.p ).toEigen();
    d.segment<3>(6)  = (a.v  - b.v ).toEigen();
    d.segment<3>(9)  = (a.bg - b.bg).toEigen();
    d.segment<3>(12) = (a.ba - b.ba).toEigen();

    /* 旋转：δφ = Log( q_b⁻¹ ⊗ q_a )
     * 与 plus 使用的右扰动严格对偶：
     *   设 a.q = b.q ⊗ Exp(δφ)  ⟹  δφ = Log( b.q⁻¹ ⊗ a.q )
     */
    d.segment<3>(3)  = q2rv((~b.q) * a.q).toEigen();

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

    const vect3 dp  (delta(0),  delta(1),  delta(2) );
    const vect3 dphi(delta(3),  delta(4),  delta(5) );
    const vect3 dv  (delta(6),  delta(7),  delta(8) );
    const vect3 dbg (delta(9),  delta(10), delta(11));
    const vect3 dba (delta(12), delta(13), delta(14));

    r.p  = s.p  + dp;
    r.v  = s.v  + dv;
    r.bg = s.bg + dbg;
    r.ba = s.ba + dba;
    r.q  = s.q * rv2q(dphi);
    r.q.normlize(&r.q);

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
    pose[0] = s.p.i;
    pose[1] = s.p.j;
    pose[2] = s.p.k;
    pose[3] = s.q.q1;   /* x */
    pose[4] = s.q.q2;   /* y */
    pose[5] = s.q.q3;   /* z */
    pose[6] = s.q.q0;   /* w */

    /* mix = {v, bg, ba [, sodo]} */
    mix[0] = s.v.i;
    mix[1] = s.v.j;
    mix[2] = s.v.k;
    mix[3] = s.bg.i;
    mix[4] = s.bg.j;
    mix[5] = s.bg.k;
    mix[6] = s.ba.i;
    mix[7] = s.ba.j;
    mix[8] = s.ba.k;
    if (with_odometer) {
        mix[9] = s.sodo;
    }
}

State fromData(const double *pose, const double *mix, bool with_odometer) {
    State s;

    s.p = vect3(pose[0], pose[1], pose[2]);

    /* 注意构造顺序：(w, x, y, z) */
    s.q = quat(pose[6], pose[3], pose[4], pose[5]);
    normlize(&s.q);

    s.v    = vect3(mix[0], mix[1], mix[2]);
    s.bg   = vect3(mix[3], mix[4], mix[5]);
    s.ba   = vect3(mix[6], mix[7], mix[8]);
    s.sodo = with_odometer ? mix[9] : 0.0;

    return s;
}


/* ============================================================================
 * 法方程累积
 * ==========================================================================*/
void accumulate_normal_equations(const mat &J, const vect &r, mat &H, vect &b) {
    assert(J.row == r.rc);

    if (H.row == 0) {
        H = mat(J.clm, J.clm, 0.0);
    }
    if (b.rc == 0) {
        b = vect(J.clm, 0.0);
    }
    assert(H.row == J.clm && H.clm == J.clm);
    assert(b.rc == J.clm);

    Eigen::MatrixXd Je = J.toEigen();
    Eigen::VectorXd re = r.toEigen();
    Eigen::MatrixXd He = H.toEigen();
    Eigen::VectorXd be = b.toEigen();

    /* H += Jᵀ J, b += Jᵀ r
     * 用 selfadjointView<Lower>().rankUpdate 更高效，但为了通用
     * （J 可能行数 > 列数，也可能列数 > 行数），直接用矩阵乘法 */
    He.noalias() += Je.transpose() * Je;
    be.noalias() += Je.transpose() * re;

    H = mat(He);
    b = vect(be);
}

/* ============================================================================
 * Schur 补
 * ==========================================================================*/
bool schur_complement(const mat &H, const vect &b, int num_marginalized,
                      mat &J_out, vect &r_out) {
    const int n = H.row;
    if (H.row != H.clm || b.rc != n || n == 0) {
        return false;
    }
    if (num_marginalized <= 0 || num_marginalized >= n) {
        return false;  /* 没有可边缘化的维，或会被边缘化光 */
    }
    const int m = num_marginalized;
    const int r = n - m;

    /* ---- 分块 ---- */
    Eigen::MatrixXd He = H.toEigen();
    He = 0.5 * (He + He.transpose()).eval();   /* 强制对称，避免数值误差 */
    Eigen::VectorXd be = b.toEigen();

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

    const mat L = llt_L(mat(H_star));
    if (L.row > 0) {
        /* H_star = L·Lᵀ，取 J_out = Lᵀ（上三角）
         * J_outᵀ r_out = b*  ⟹  L·r_out = b*  ⟹  r_out = L⁻¹·b* */
        J_out_e = L.toEigen().transpose();
        const Eigen::MatrixXd Le = L.toEigen();
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

    J_out = mat(J_out_e);
    r_out = vect(r_out_e);
    return true;
}

/* ============================================================================
 * 高层封装
 * ==========================================================================*/
void MarginalizationInfo::reset() {
    n_ = 0;
    H_ = mat();
    b_ = vect();
    J_prior_ = mat();
    r_prior_ = vect();
}

void MarginalizationInfo::addResidual(const mat &J, const vect &r) {
    accumulate_normal_equations(J, r, H_, b_);
    n_ = H_.row;
}

void MarginalizationInfo::setPrior(const mat &J_out, const vect &r_out) {
    J_prior_ = J_out;
    r_prior_ = r_out;
    /* 先验法方程 H += J_outᵀ J_out, b += J_outᵀ r_out */
    accumulate_normal_equations(J_out, r_out, H_, b_);
    n_ = H_.row;
}

bool MarginalizationInfo::marginalize(int num_marginalized) {
    if (n_ == 0) {
        return false;
    }
    mat J_new;
    vect r_new;
    if (!schur_complement(H_, b_, num_marginalized, J_new, r_new)) {
        return false;
    }
    /* 边缘化完成后，把结果保存为先验，并让累积状态回到"只有先验"的状态 */
    J_prior_ = J_new;
    r_prior_ = r_new;
    H_ = mat();
    b_ = vect();
    accumulate_normal_equations(J_prior_, r_prior_, H_, b_);
    n_ = H_.row;
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
            vo.xyz_c   = vect3(tf.xyz_c(0), tf.xyz_c(1), 1.0);
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
                                  const vect3 &g_n) {
    /* ---- 0) 标定参数块（全局，窗口内共享） ---- */
    problem.AddParameterBlock(calib_data_.data(), NUM_CALIB);
    problem.SetManifold(calib_data_.data(), new CalibManifold());
    if (!opt_.calib_mode) {
        problem.SetParameterBlockConstant(calib_data_.data());   /* 导航模式：固定 */
    }

    /* ---- 1) 帧参数块 + Manifold ---- */
    for (auto &f : frames) {
        problem.AddParameterBlock(f.pose.data(), NUM_POSE);
        problem.SetManifold(f.pose.data(), new PoseManifold());
        problem.AddParameterBlock(f.mix.data(), NUM_MIX_ODO);
    }
    if (opt_.fix_first_pose && !frames.empty()) {
        problem.SetParameterBlockConstant(frames.front().pose.data());
    }

    /* ---- 2) 预积分因子 ---- */
    for (size_t k = 0; k + 1 < frames.size(); ++k) {
        if (!frames[k].pre_to_next) continue;
        auto *cost = new ceres::AutoDiffCostFunction<PreintResidual,
                                                     15, 7, 10, 7, 10>(
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
        auto *cost = new ceres::AutoDiffCostFunction<GnssVelResidual, 3, 10>(
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
        auto *cost = new ceres::AutoDiffCostFunction<StaticVelResidual, 3, 10>(
            new StaticVelResidual(opt_.static_vel_sigma, 1.0));
        problem.AddResidualBlock(cost, nullptr, f.mix.data());
    }

    /* ---- 7) 里程计速度因子（引用整个标定块） ---- */
    for (auto &f : frames) {
        if (!f.has_odo) continue;
        auto *cost = new ceres::AutoDiffCostFunction<OdoVelResidual, 3, 7, 10, NUM_CALIB>(
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

            problem.AddParameterBlock(inv_depth, 1);
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
}

/* ============================================================================
 * GraphOptimizer::marginalizeOldestFrame
 *
 *   说明：真正的边缘化需要"逐残差"地提取 J 与 r。Ceres 没有直接暴露
 *   "给定 ResidualBlockId 求雅可比"的接口，因此实现上分两步：
 *     (a) 先把 frame[0] 相关的残差块临时从 problem 里移除；
 *     (b) 用 ceres::Problem::Evaluate 在"当前参数点"处求所有残差的
 *         J 与 r，累积成法方程，再做 Schur 补。
 *
 *   本示例只做 (b) 的简化：调用方保证所有 frame[0] 相关残差都在
 *   problem 里；若需精确的分块提取，建议在因子层追加
 *   "GetLinearizationData" 接口，由因子自己返回 J/r。
 * ==========================================================================*/
bool GraphOptimizer::marginalizeOldestFrame(ceres::Problem &problem,
                                            std::vector<Frame> &frames,
                                            const mat  &J_prior_old,
                                            const vect &r_prior_old) {
    if (frames.size() < 2) return false;

    /* ---- 1) 用 Problem::Evaluate 求全局 J 与 r ---- */
    ceres::Problem::EvaluateOptions eval_opt;
    eval_opt.apply_loss_function = false;

    double cost = 0.0;
    std::vector<double> residuals;
    std::vector<double>  gradient;          // ← 第 4 个参数（梯度过期不用）
    ceres::CRSMatrix jacobian;

    if (!problem.Evaluate(eval_opt, &cost, &residuals, &gradient, &jacobian)) {
        return false;
    }

    /* 把 CRSMatrix 展成稠密矩阵 */
    const int n_rows = jacobian.num_rows;
    const int n_cols = jacobian.num_cols;
    Eigen::MatrixXd J_full = Eigen::MatrixXd::Zero(n_rows, n_cols);
    for (int i = 0; i < n_rows; ++i) {
        for (int k = jacobian.rows[i]; k < jacobian.rows[i + 1]; ++k) {
            J_full(i, jacobian.cols[k]) = jacobian.values[k];
        }
    }
    Eigen::VectorXd r_full(residuals.size());
    for (size_t i = 0; i < residuals.size(); ++i) r_full(i) = residuals[i];

    /* ---- 2) 组装法方程 ---- */
    mat  H_acc;
    vect b_acc;
    if (J_prior_old.row > 0) {
        /* 先验也是残差之一：H = J_pᵀ J_p, b = J_pᵀ r_p */
        accumulate_normal_equations(J_prior_old, r_prior_old, H_acc, b_acc);
    }

    /* 全问题 J/r 累积 */
    accumulate_normal_equations(mat(J_full), vect(r_full), H_acc, b_acc);

    /* ---- 3) Schur 补：消去 frame[0] 的全部切空间变量 ---- */
    /* 注意：切空间布局取决于 Ceres 内部的参数块顺序与切空间维度。
     * 第一帧贡献 6 (pose) + 10 (mix) = 16 维。
     * 但如果 fix_first_pose 为真，第一帧 pose 是常量，不参与切空间，
     * 那么只需要消去 mix 的 10 维；这里假设 fix_first_pose=false，
     * 通用地消去 16 维。实际工程中请按 Ceres 的 variable ordering
     * 精确映射。 */
    const int dim_marg = 16;
    mat  J_new;
    vect r_new;
    if (!schur_complement(H_acc, b_acc, dim_marg, J_new, r_new)) {
        return false;
    }

    J_prior_ = J_new;
    r_prior_ = r_new;
    return true;
}

/* ============================================================================
 * GraphOptimizer::optimize
 * ==========================================================================*/
bool GraphOptimizer::optimize(std::vector<Frame> &frames, const vect3 &g_n) {
    if (frames.empty()) return false;

    /* ---- 1) 组装问题 ---- */
    ceres::Problem problem;
    buildProblem(problem, frames, g_n);

    /* 若已有先验，作为参数块加入（这里用简化版：把先验视作一个
     * 覆盖"所有保留变量"的大残差块。实际项目应按变量分块追加）。 */
    if (J_prior_.row > 0) {
        /* 注意：下面的调用需要把 J_prior_ 的列映射到 frame 参数块的指针，
         * 这里只演示接口，实际使用时请按变量顺序展开。 */
        // problem.AddResidualBlock(
        //     new MarginalizationPriorFactor(J_prior_, r_prior_), nullptr,
        //     /* 依次列出所有被覆盖的参数块 */ ...);
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
static vect3 blhToNeu(const LocalFrame &lf, const DataSensor281_t &s) {
    return lf.blh2neu(vect3(s.posgps[0], s.posgps[1], s.posgps[2]));
}

/* 由双天线航向 + 水平，构造初始姿态 --------------------------------- */
static quat initAttitudeFromYaw(double yaw) {
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
    lf.init(vect3(raw[first_gnss].posgps[0],
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
        calib.lvOD.i == 0.0 && calib.lvOD.j == 0.0 && calib.lvOD.k == 0.0 &&
        calib.yaw_gnss_offset == 0.0) {
        calib.sodo           = opt.odo_sodo_init;
        calib.abv_pitch      = opt.odo_abv.i;
        calib.abv_yaw        = opt.odo_abv.j;
        calib.lvOD           = opt.odo_lvOD;
        calib.yaw_gnss_offset = opt.yaw_offset;
    }
    gopt.setCalibState(calib);

    auto evalFrameCompare = [&gopt](const Frame &f) {
        FrameCompare c;
        c.time = f.time;

        /* --- 位置 --- */
        if (f.has_gnss) {
            c.has_gnss_pos = true;
            const double pe = f.pose[0] - f.gnss_pos.i;
            const double pn = f.pose[1] - f.gnss_pos.j;
            const double pu = f.pose[2] - f.gnss_pos.k;
            c.dpe = pe; c.dpn = pn; c.dpu = pu;
            c.dp_xy = std::sqrt(pe * pe + pn * pn);
            c.dp_3d = std::sqrt(pe * pe + pn * pn + pu * pu);
        }

        /* --- 速度 --- */
        if (f.has_gnss_vel) {
            c.has_gnss_vel = true;
            const double ve = f.mix[0] - f.gnss_vn.i;
            const double vn = f.mix[1] - f.gnss_vn.j;
            const double vu = f.mix[2] - f.gnss_vn.k;
            c.dve = ve; c.dvn = vn; c.dvu = vu;
            c.dv_xy = std::sqrt(ve * ve + vn * vn);
            c.dv_3d = std::sqrt(ve * ve + vn * vn + vu * vu);
        }

        /* --- 航向：raw_yaw_conv 只含 ant_mode 折算，需叠加 yaw_off_calib --- */
        if (f.has_raw_yaw) {
            const double yoff = gopt.getCalibState().yaw_gnss_offset;  // ← 实时
            c.has_gnss_yaw = true;
            quat q(f.pose[6], f.pose[3], f.pose[4], f.pose[5]);
            double yaw_est = q2att(q).k;
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
        vect3 p(f.pose[0], f.pose[1], f.pose[2]);
        double e = norm(p - f.gnss_pos);
        acc.add(f.time, e);
    };

    std::printf("[runner] %s模式 | 标定初值: sodo=%.4f, abv=(%.2f, %.2f)°, lvOD=(%.3f, %.3f, %.3f), yaw_off=%.3f°\n",
                opt.calib_mode ? "标定" : "导航",
                calib.sodo,
                calib.abv_pitch / DEG, calib.abv_yaw / DEG,
                calib.lvOD.i, calib.lvOD.j, calib.lvOD.k,
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
        quat q0 = initAttitudeFromYaw(yaw0);
        std::printf("[runner] 初始航向: %.3f deg (%-5s, 双天线原始 %.3f deg)\n",
                    yaw0 / DEG, used_gnss_yaw0 ? "GNSS" : "默认0",
                    raw[first_gnss].yaw / DEG);

        f0.pose = { 0, 0, 0, q0.q1, q0.q2, q0.q3, q0.q0 };
        f0.mix = { 0, 0, 0, 0, 0, 0, 0, 0, 0, opt.odo_sodo_init };
        f0.has_gnss = true;
        f0.gnss_pos = vect3(0, 0, 0);   /* 局部系原点 */
        f0.gnss_std = vect3(opt.gnss_pos_scale * std::max(raw[first_gnss].posstd[0], 0.5),
                            opt.gnss_pos_scale * std::max(raw[first_gnss].posstd[1], 0.5),
                            opt.gnss_pos_scale * std::max(raw[first_gnss].posstd[2], 1.0));
        frames.push_back(f0);
        st.n_gnss_used++;
    }

    PreintegrationParam pp;
    pp.gyr_arw      = 0.5 * DEG / 60.0;    /* 0.5 deg/sqrt(hr) */
    pp.acc_vrw      = 0.1 * 1e-3;          /* 0.1 mg/sqrt(Hz) → 0.1e-3 m/s^1.5 近似 */
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
    vect3  bg_cur(O31), ba_cur(O31);     /*< 下一段预积分用的零偏 */

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
        m.dtheta = vect3(s.wm[0], s.wm[1], s.wm[2]);
        m.dvel   = vect3(s.vm[0], s.vm[1], s.vm[2]);
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

        const vect3 dv_b0 = preint.v();
        const quat   dq_b0 = preint.q();
        const vect3  dp_b0 = preint.p();
        const double T_k   = preint.dt();

        quat q_prev(fprev.pose[6], fprev.pose[3], fprev.pose[4], fprev.pose[5]);
        vect3 v_prev(fprev.mix[0], fprev.mix[1], fprev.mix[2]);
        vect3 p_prev(fprev.pose[0], fprev.pose[1], fprev.pose[2]);

        quat q_pred = q_prev * dq_b0;
        normlize(&q_pred);
        Eigen::Matrix3d R_prev = q_prev.toEigen().toRotationMatrix();
        Eigen::Vector3d g_n = lf.earthModel().gn.toEigen();
        Eigen::Vector3d v_pred = v_prev.toEigen()
                               + R_prev * dv_b0.toEigen()
                               + g_n * T_k;
        Eigen::Vector3d p_pred = p_prev.toEigen()
                               + v_prev.toEigen() * T_k
                               + 0.5 * g_n * T_k * T_k
                               + R_prev * dp_b0.toEigen();

        Frame fk;
        fk.time = s.t;
        fk.pose = { p_pred(0), p_pred(1), p_pred(2),
                    q_pred.q1, q_pred.q2, q_pred.q3, q_pred.q0 };
        fk.mix  = { v_pred(0), v_pred(1), v_pred(2),
                    fprev.mix[3], fprev.mix[4], fprev.mix[5],
                    fprev.mix[6], fprev.mix[7], fprev.mix[8], fprev.mix[9] };

        /* 4.5 观测填充 */
        if (has_gnss_now) {
            fk.has_gnss = true;
            fk.gnss_pos = blhToNeu(lf, s);
            fk.gnss_std = vect3(opt.gnss_pos_scale * std::max(s.posstd[0], 0.5),
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
                fk.gnss_vn     = vect3(s.vngps[0], s.vngps[1], s.vngps[2]);
                fk.gnss_vn_std = vect3(opt.gnss_vel_scale * 0.1 * sat_scale,
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
            fk.odo_omega_meas = vect3(s.wm[0] / dt, s.wm[1] / dt, s.wm[2] / dt);
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
                q2att(quat(fk.pose[6], fk.pose[3], fk.pose[4], fk.pose[5])).k / DEG,
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
                            norm(vect3(fg.pose[0], fg.pose[1], fg.pose[2]) - fg.gnss_pos));
            }

            /* 优化后、erase 之前，保存最旧帧到完整轨迹 */
            trajectory.push_back(frames.front());   // ← 新增
            frames.erase(frames.begin());
        }

        /* 4.11 把最新零偏估计回灌给下一段预积分
         *      预积分在积分时就把零偏扣掉，外推出的初值才是零偏补偿后的结果；
         *      它同时是因子中零偏一阶展开的线性化点。 */
        if (solved && !frames.empty()) {
            const Frame &fnew = frames.back();
            bg_cur = vect3(fnew.mix[3], fnew.mix[4], fnew.mix[5]);
            ba_cur = vect3(fnew.mix[6], fnew.mix[7], fnew.mix[8]);
            preint.setBias(bg_cur, ba_cur);
        }
    }

    /* ---- 5) 收尾：对最后一批关键帧跑一次优化 ---- */
    if (!frames.empty()) {
        const vect3 g_vect = lf.earthModel().gn;
        gopt.optimize(frames, g_vect);

        /* 末尾残留帧也加入轨迹 */
        for (const auto &f : frames) {
            evalOneFrame(f, acc_opt);                /* ← 新增 */
            if (f.gnss_in_gap) {
                acc_gap.add(f.time, norm(vect3(f.pose[0], f.pose[1], f.pose[2]) - f.gnss_pos));
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
                st.calib_final.lvOD.i, st.calib_final.lvOD.j, st.calib_final.lvOD.k);
    std::printf("  航向偏置        : %+10.4f deg\n", st.calib_final.yaw_gnss_offset / DEG);

    st.n_kf       = static_cast<int>(frames.size());
    st.total_time = raw.back().t - raw.front().t;
    st.final_cost = gopt.hasPrior() ? gopt.r_prior().toEigen().norm() : 0.0;

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
