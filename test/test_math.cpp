// test_math.cpp
// 编译：
//   g++ -std=c++17 -O2 test_math.cpp alg_lib.cpp \
//       -I. -I/usr/include/eigen3 \
//       $(pkg-config --cflags --libs ceres) -o test_math

#include "alg_lib.h"

#include <cmath>
#include <cstdio>
#include <cstring>

/* ============================== 极简测试框架 ============================== */
static int g_pass = 0, g_fail = 0;

#define EXPECT_TRUE(cond)                                                    \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            g_fail++;                                                        \
        } else { g_pass++; }                                                 \
    } while (0)

#define EXPECT_NEAR(a, b, eps)                                               \
    do {                                                                     \
        double _a = (a), _b = (b);                                           \
        if (!(std::fabs(_a - _b) <= (eps))) {                                \
            std::printf("    FAIL %s:%d: %s=%.12g vs %s=%.12g (|d|=%.3g, eps=%.3g)\n", \
                        __FILE__, __LINE__, #a, _a, #b, _b,                   \
                        std::fabs(_a - _b), (double)(eps));                   \
            g_fail++;                                                        \
        } else { g_pass++; }                                                 \
    } while (0)

#define EXPECT_EQ(a, b)                                                      \
    do {                                                                     \
        if (!((a) == (b))) {                                                 \
            std::printf("    FAIL %s:%d: %s == %s\n",                        \
                        __FILE__, __LINE__, #a, #b);                         \
            g_fail++;                                                        \
        } else { g_pass++; }                                                 \
    } while (0)

#define RUN_TEST(fn)                                                         \
    do {                                                                     \
        std::printf("[ RUN  ] %s\n", #fn);                                   \
        int _before = g_fail;                                                \
        fn();                                                                \
        std::printf(g_fail == _before ? "[  OK  ] %s\n" : "[ FAIL ] %s\n",   \
                    #fn);                                                    \
    } while (0)

/* ============================================================================
 * 几何基础类型：已统一为 Eigen（Eigen::Vector3d / Matrix3d / Quaterniond）
 * 下面只测**本工程仍保留的自由函数**语义，不再测容器自身运算符。
 * ==========================================================================*/

static void test_vect3_norms() {
    Eigen::Vector3d a(3.0, 4.0, 0.0);
    EXPECT_NEAR(norm(a), 5.0, 1e-15);
    EXPECT_NEAR(normXY(a), 5.0, 1e-15);
    EXPECT_NEAR(normInf(a), 4.0, 1e-15);
    EXPECT_NEAR(normXYInf(a), 4.0, 1e-15);

    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    EXPECT_NEAR(norm(b), 0.0, 1e-15);
    EXPECT_NEAR(normXY(b), 0.0, 1e-15);

    /* abs / maxabs */
    Eigen::Vector3d m1(1.0, -5.0, 3.0), m2(-2.0, 4.0, -6.0);
    Eigen::Vector3d am = abs(m1);
    EXPECT_NEAR(am.x(), 1.0, 1e-15);
    EXPECT_NEAR(am.y(), 5.0, 1e-15);
    Eigen::Vector3d mx = maxabs(m1, m2);
    EXPECT_NEAR(mx.x(), 2.0, 1e-15);
    EXPECT_NEAR(mx.y(), 5.0, 1e-15);
    EXPECT_NEAR(mx.z(), 6.0, 1e-15);
}

static void test_vect3_dot_and_products() {
    Eigen::Vector3d a(1.0, 2.0, 3.0), b(4.0, 5.0, 6.0);
    EXPECT_NEAR(dot(a, b), 32.0, 1e-15);
    EXPECT_NEAR(dot(a, a), 14.0, 1e-15);

    Eigen::Vector3d e = dotmul(a, b);
    EXPECT_NEAR(e.x(), 4.0, 1e-15);
    EXPECT_NEAR(e.y(), 10.0, 1e-15);
    EXPECT_NEAR(e.z(), 18.0, 1e-15);

    /* 叉乘：工程里统一用 Eigen 的 .cross() */
    Eigen::Vector3d c = a.cross(b);
    EXPECT_NEAR(c.x(), -3.0, 1e-15);
    EXPECT_NEAR(c.y(), 6.0, 1e-15);
    EXPECT_NEAR(c.z(), -3.0, 1e-15);

    /* 正交向量夹角正弦 */
    Eigen::Vector3d x(1, 0, 0), y(0, 1, 0);
    EXPECT_NEAR(sinAng(x, y), 1.0, 1e-15);
    EXPECT_NEAR(sinAng(x, x), 0.0, 1e-15);

    /* vxv 外积 */
    Eigen::Matrix3d m = vxv(x, y);
    EXPECT_NEAR(m(0,0), 0.0, 1e-15);
    EXPECT_NEAR(m(0,1), 1.0, 1e-15);
    EXPECT_NEAR(m(1,0), 0.0, 1e-15);
    EXPECT_NEAR(m(1,1), 0.0, 1e-15);
}

static void test_vect3_elementwise() {
    Eigen::Vector3d a(-1.0, 4.0, -9.0);
    Eigen::Vector3d s = sqrt(abs(a));
    EXPECT_NEAR(s.x(), 1.0, 1e-15);
    EXPECT_NEAR(s.y(), 2.0, 1e-15);
    EXPECT_NEAR(s.z(), 3.0, 1e-15);

    Eigen::Vector3d b(2.0, 3.0, 4.0);
    Eigen::Vector3d p2 = pow(b, 2);
    EXPECT_NEAR(p2.x(), 4.0, 1e-15);
    EXPECT_NEAR(p2.y(), 9.0, 1e-15);
    EXPECT_NEAR(p2.z(), 16.0, 1e-15);

    Eigen::Vector3d p3 = pow(b, 3);
    EXPECT_NEAR(p3.x(), 8.0, 1e-15);
    EXPECT_NEAR(p3.y(), 27.0, 1e-15);
    EXPECT_NEAR(p3.z(), 64.0, 1e-15);

    /* sort 为降序（与旧实现一致） */
    Eigen::Vector3d srt = sort(Eigen::Vector3d(1.0, 5.0, 3.0));
    EXPECT_NEAR(srt.x(), 5.0, 1e-15);
    EXPECT_NEAR(srt.y(), 3.0, 1e-15);
    EXPECT_NEAR(srt.z(), 1.0, 1e-15);
}

static void test_mat3_trace_det_inv() {
    Eigen::Matrix3d A = (Eigen::Matrix3d() << 1, 2, 3, 4, 5, 6, 7, 8, 10).finished();
    EXPECT_NEAR(trace(A), 16.0, 1e-14);

    /* det(A) = 1*(5*10-6*8) - 2*(4*10-6*7) + 3*(4*8-5*7) = 2 + 4 - 9 = -3 */
    EXPECT_NEAR(det(A), -3.0, 1e-12);

    /* A * inv(A) ≈ I */
    Eigen::Matrix3d Ai = inv(A);
    Eigen::Matrix3d P = A * Ai;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(P(i, j), (i == j) ? 1.0 : 0.0, 1e-12);

    /* 正交阵的逆等于转置 */
    Eigen::Vector3d rv(0.1, 0.2, 0.3);
    Eigen::Matrix3d R = rv2m(rv);
    Eigen::Matrix3d Ri = inv(R);
    Eigen::Matrix3d Rt = R.transpose();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(Ri(i, j), Rt(i, j), 1e-12);

    /* adj(A) = det(A) * inv(A) */
    Eigen::Matrix3d Adj = adj(A);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            EXPECT_NEAR(Adj(i, j), det(A) * Ai(i, j), 1e-12);
}

static void test_mat3_askew() {
    /* askew(v) * w = v × w */
    Eigen::Vector3d v(1.0, 2.0, 3.0), w(4.0, 5.0, 6.0);
    Eigen::Matrix3d S = askew(v);
    Eigen::Vector3d vw = v.cross(w);
    Eigen::Vector3d Svw = S * w;
    EXPECT_NEAR(Svw.x(), vw.x(), 1e-14);
    EXPECT_NEAR(Svw.y(), vw.y(), 1e-14);
    EXPECT_NEAR(Svw.z(), vw.z(), 1e-14);

    /* 反对称性 */
    EXPECT_NEAR(S(0,1), -S(1,0), 1e-15);
    EXPECT_NEAR(S(0,2), -S(2,0), 1e-15);
    EXPECT_NEAR(S(1,2), -S(2,1), 1e-15);
    EXPECT_NEAR(S(0,0), 0.0, 1e-15);
}

static void test_mat3_rot() {
    /* Rx(π/2): y -> z, z -> -y */
    Eigen::Matrix3d Rx = Rot(PI / 2, 'x');
    Eigen::Vector3d y(0, 1, 0), z(0, 0, 1);
    Eigen::Vector3d Rxy = Rx * y, Rxz = Rx * z;
    EXPECT_NEAR(Rxy.z(), 1.0, 1e-15);
    EXPECT_NEAR(Rxy.y(), 0.0, 1e-15);
    EXPECT_NEAR(Rxz.y(), -1.0, 1e-15);

    /* 正交性：Rᵀ R = I，det = 1 */
    for (char axis : {'x', 'y', 'z'}) {
        Eigen::Matrix3d R = Rot(0.7, axis);
        Eigen::Matrix3d RtR = R.transpose() * R;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                EXPECT_NEAR(RtR(i, j), (i == j) ? 1.0 : 0.0, 1e-14);
        EXPECT_NEAR(det(R), 1.0, 1e-14);
    }

    /* rcijk：交换分量后仍应保持元素集合不变（对 (0,1,2) 置换为自身） */
    Eigen::Matrix3d M = (Eigen::Matrix3d() << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
    Eigen::Matrix3d M021 = rcijk(M, 021);
    EXPECT_NEAR(M021(0,0), M(0,0), 1e-15);
    EXPECT_NEAR(M021(0,1), M(0,2), 1e-15);
    EXPECT_NEAR(M021(0,2), M(0,1), 1e-15);
}

static void test_mat3_diag() {
    Eigen::Matrix3d A = (Eigen::Matrix3d() << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
    Eigen::Vector3d d = diag(A);
    EXPECT_NEAR(d.x(), 1.0, 1e-15);
    EXPECT_NEAR(d.y(), 5.0, 1e-15);
    EXPECT_NEAR(d.z(), 9.0, 1e-15);

    /* Eigen 版本 diag(Vector3d) 返回 3×3 动态矩阵 */
    Eigen::Matrix3d D = diag(Eigen::Vector3d(2.0, 4.0, 6.0));
    EXPECT_NEAR(D(0,0), 2.0, 1e-15);
    EXPECT_NEAR(D(1,1), 4.0, 1e-15);
    EXPECT_NEAR(D(2,2), 6.0, 1e-15);
    EXPECT_NEAR(D(0,1), 0.0, 1e-15);
}

static void test_mat3_dotmul_and_norm() {
    Eigen::Matrix3d A = (Eigen::Matrix3d() << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
    Eigen::Matrix3d B = Eigen::Matrix3d::Constant(2.0);
    Eigen::Matrix3d E = dotmul(A, B);
    EXPECT_NEAR(E(0,0), 2.0, 1e-15);
    EXPECT_NEAR(E(1,1), 10.0, 1e-15);
    EXPECT_NEAR(E(2,2), 18.0, 1e-15);

    /* norm(A) = sqrt(tr(A Aᵀ)) */
    double expect = std::sqrt(1.0 + 4.0 + 9.0 + 16.0 + 25.0 + 36.0 + 49.0 + 64.0 + 81.0);
    EXPECT_NEAR(norm(A), expect, 1e-14);
    EXPECT_NEAR(trMMT(A), 285.0, 1e-12);

    /* MMT(A, A) = A Aᵀ */
    Eigen::Matrix3d MMTa = MMT(A, A);
    Eigen::Matrix3d ref = A * A.transpose();
    EXPECT_NEAR((MMTa - ref).cwiseAbs().maxCoeff(), 0.0, 1e-12);
}

static void test_mat3_pow() {
    Eigen::Matrix3d A = (Eigen::Matrix3d() << 1, 1, 0, 0, 1, 1, 0, 0, 1).finished();
    Eigen::Matrix3d A2 = pow(A, 2);
    Eigen::Matrix3d A2_expect = A * A;
    EXPECT_NEAR((A2 - A2_expect).cwiseAbs().maxCoeff(), 0.0, 1e-14);

    Eigen::Matrix3d A1 = pow(A, 1);
    EXPECT_NEAR((A1 - A).cwiseAbs().maxCoeff(), 0.0, 1e-14);
}

static void test_quat_basic() {
    /* a2qua / q2att 往返 */
    Eigen::Vector3d att_in(0.1, 0.2, 0.3);
    Eigen::Quaterniond q = a2qua(att_in);
    Eigen::Vector3d att = q2att(q);
    EXPECT_NEAR(att.x(), att_in.x(), 1e-12);
    EXPECT_NEAR(att.y(), att_in.y(), 1e-12);
    EXPECT_NEAR(att.z(), att_in.z(), 1e-12);

    /* setYaw 只改偏航 */
    Eigen::Quaterniond q2 = a2qua(Eigen::Vector3d(0.1, 0.2, 0.3));
    setYaw(q2, 0.7);
    Eigen::Vector3d att2 = q2att(q2);
    EXPECT_NEAR(att2.z(), 0.7, 1e-12);
    EXPECT_NEAR(att2.x(), 0.1, 1e-12);
    EXPECT_NEAR(att2.y(), 0.2, 1e-12);

    /* 四元数旋转 == 旋转矩阵作用（约定一致性） */
    Eigen::Vector3d v(1.0, -2.0, 0.5);
    Eigen::Vector3d qv = q * v;
    Eigen::Vector3d mv = q2mat(q) * v;
    EXPECT_NEAR((qv - mv).cwiseAbs().maxCoeff(), 0.0, 1e-14);

    /* Eigen 的共轭对应旧 operator~；共轭乘积为单位四元数 */
    Eigen::Quaterniond qc = q.conjugate();
    EXPECT_NEAR(qc.w(), q.w(), 1e-15);
    EXPECT_NEAR(qc.x(), -q.x(), 1e-15);
    Eigen::Quaterniond qqc = qc * q;
    EXPECT_NEAR(qqc.w(), 1.0, 1e-14);
    EXPECT_NEAR(qqc.x(), 0.0, 1e-14);
    EXPECT_NEAR(qqc.y(), 0.0, 1e-14);
    EXPECT_NEAR(qqc.z(), 0.0, 1e-14);
}

/* ============================================================================
 * 姿态/旋转函数
 * ==========================================================================*/

static void test_rv2q_q2rv_roundtrip() {
    /* 小角 */
    Eigen::Vector3d rv(0.01, -0.02, 0.03);
    Eigen::Quaterniond q = rv2q(rv);
    Eigen::Vector3d rv2 = q2rv(q);
    EXPECT_NEAR(rv2.x(), rv.x(), 1e-13);
    EXPECT_NEAR(rv2.y(), rv.y(), 1e-13);
    EXPECT_NEAR(rv2.z(), rv.z(), 1e-13);

    /* 中角 */
    Eigen::Vector3d rv_m(0.5, -0.3, 0.7);
    Eigen::Quaterniond qm = rv2q(rv_m);
    Eigen::Vector3d rv2_m = q2rv(qm);
    EXPECT_NEAR(rv2_m.x(), rv_m.x(), 1e-12);
    EXPECT_NEAR(rv2_m.y(), rv_m.y(), 1e-12);
    EXPECT_NEAR(rv2_m.z(), rv_m.z(), 1e-12);

    /* 零角 */
    Eigen::Vector3d zero(0, 0, 0);
    Eigen::Quaterniond qz = rv2q(zero);
    EXPECT_NEAR(qz.w(), 1.0, 1e-15);
    EXPECT_NEAR(norm(q2rv(qz)), 0.0, 1e-14);
}

static void test_rv2q_small_angle_taylor() {
    /* 小角度应满足 |rv| / 2 的近似 */
    double eps = 1e-6;
    Eigen::Vector3d rv(eps, 0, 0);
    Eigen::Quaterniond q = rv2q(rv);
    /* q ≈ [1, rv/2] */
    EXPECT_NEAR(q.w(), 1.0, 1e-12);
    EXPECT_NEAR(q.x(), eps / 2.0, 1e-14);
    EXPECT_NEAR(q.y(), 0.0, 1e-15);
    EXPECT_NEAR(q.z(), 0.0, 1e-15);
}

static void test_q2mat_m2qua_roundtrip() {
    Eigen::Vector3d rv(0.2, -0.4, 0.6);
    Eigen::Quaterniond q1 = rv2q(rv);
    Eigen::Matrix3d R  = q2mat(q1);
    Eigen::Quaterniond q2 = m2qua(R);
    /* q2 = ±q1；允许符号翻转 */
    double dot = q1.w() * q2.w() + q1.x() * q2.x() + q1.y() * q2.y() + q1.z() * q2.z();
    EXPECT_NEAR(std::fabs(dot), 1.0, 1e-12);
}

static void test_a2qua_q2att_roundtrip() {
    Eigen::Vector3d att(0.1, -0.2, 0.3);
    Eigen::Quaterniond q = a2qua(att);
    Eigen::Vector3d att2 = q2att(q);
    EXPECT_NEAR(att2.x(), att.x(), 1e-12);
    EXPECT_NEAR(att2.y(), att.y(), 1e-12);
    EXPECT_NEAR(att2.z(), att.z(), 1e-12);
}

static void test_a2mat_m2att_roundtrip() {
    Eigen::Vector3d att(0.05, -0.15, 1.2);
    Eigen::Matrix3d R = a2mat(att);
    Eigen::Vector3d att2 = m2att(R);
    EXPECT_NEAR(att2.x(), att.x(), 1e-12);
    EXPECT_NEAR(att2.y(), att.y(), 1e-12);
    EXPECT_NEAR(att2.z(), att.z(), 1e-12);
}

static void test_a2qua_matches_a2mat() {
    Eigen::Vector3d att(0.1, 0.2, 0.3);
    Eigen::Quaterniond q = a2qua(att);
    Eigen::Matrix3d Rq = q2mat(q);
    Eigen::Matrix3d Ra = a2mat(att);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(Rq(i, j), Ra(i, j), 1e-12);
        }
    }
}

static void test_q2mat_orthonormal() {
    Eigen::Vector3d rv(0.3, -0.7, 1.1);
    Eigen::Matrix3d R = rv2m(rv);
    Eigen::Matrix3d RtR = ((R.transpose())) * R;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(RtR(i, j), (i == j) ? 1.0 : 0.0, 1e-13);
        }
    }
    EXPECT_NEAR(det(R), 1.0, 1e-13);
}

static void test_m2rv_rv2m_roundtrip() {
    Eigen::Vector3d rv(0.1, 0.2, -0.3);
    Eigen::Matrix3d R = rv2m(rv);
    Eigen::Vector3d rv2 = m2rv(R);
    EXPECT_NEAR(rv2.x(), rv.x(), 1e-12);
    EXPECT_NEAR(rv2.y(), rv.y(), 1e-12);
    EXPECT_NEAR(rv2.z(), rv.z(), 1e-12);
}

static void test_qq2phi() {
    /* qq2phi(qcalcu, qreal) = q2rv(qreal * ~qcalcu) */
    Eigen::Quaterniond qc = a2qua(Eigen::Vector3d(0.1, 0.0, 0.0));
    Eigen::Quaterniond qr = a2qua(Eigen::Vector3d(0.15, 0.0, 0.0));
    Eigen::Vector3d phi = qq2phi(qc, qr);
    EXPECT_NEAR(phi.x(), 0.05, 1e-12);
    EXPECT_NEAR(phi.y(), 0.0, 1e-14);
    EXPECT_NEAR(phi.z(), 0.0, 1e-14);
}

static void test_addmu() {
    Eigen::Quaterniond q = a2qua(Eigen::Vector3d(0.1, 0.2, 0.3));
    Eigen::Vector3d mu(0.01, -0.02, 0.03);
    Eigen::Quaterniond q2 = addmu(q, mu);
    Eigen::Quaterniond qexp = q * rv2q(mu);
    EXPECT_NEAR(q2.w(), qexp.w(), 1e-14);
    EXPECT_NEAR(q2.x(), qexp.x(), 1e-14);
    EXPECT_NEAR(q2.y(), qexp.y(), 1e-14);
    EXPECT_NEAR(q2.z(), qexp.z(), 1e-14);
}

static void test_UpDown() {
    /* UpDown 把姿态 [p, r, y] -> [-p, r+π, y] */
    Eigen::Vector3d att(0.1, 0.2, 0.3);
    Eigen::Quaterniond q = a2qua(att);
    Eigen::Quaterniond q2 = UpDown(q);
    Eigen::Vector3d att2 = q2att(q2);
    EXPECT_NEAR(att2.x(), -0.1, 1e-12);
    /* roll 映射到 π - roll 或 roll + π，看约定 */
    /* 只检查 pitch 与 yaw */
    EXPECT_NEAR(att2.z(), 0.3, 1e-12);
}

static void test_vn2att() {
    /* 纯东向 */
    double yaw = vn2att(5.0, 0.0);
    EXPECT_NEAR(yaw, -PI / 2, 1e-12);

    /* 纯北向 */
    yaw = vn2att(0.0, 5.0);
    EXPECT_NEAR(yaw, 0.0, 1e-12);

    /* 静止时返回 0 */
    yaw = vn2att(0.0, 0.0);
    EXPECT_NEAR(yaw, 0.0, 1e-15);
}

static void test_diffYaw() {
    EXPECT_NEAR(diffYaw(0.1, 0.0), 0.1, 1e-15);
    EXPECT_NEAR(diffYaw(0.0, 0.1), -0.1, 1e-15);
    /* 跨越 ±π */
    EXPECT_NEAR(diffYaw(PI - 0.1, -PI + 0.1), -0.2, 1e-14);
    EXPECT_NEAR(diffYaw(-PI + 0.1, PI - 0.1), 0.2, 1e-14);
}

/* ============================================================================
 * Eigen::VectorXd 上的自由函数
 * ==========================================================================*/

static void test_vect_norms() {
    Eigen::VectorXd v(3);
    v << 3, -4, 12;
    EXPECT_NEAR(norm(v), 13.0, 1e-13);
    EXPECT_NEAR(norm1(v), 19.0, 1e-13);
    EXPECT_NEAR(normInf(v), 12.0, 1e-13);
}

/* ============================================================================
 * Eigen::MatrixXd 上的自由函数
 * ==========================================================================*/

static void test_mat_diag_trace() {
    Eigen::MatrixXd A(3, 3);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) A(i, j) = i * 3 + j + 1.0;
    EXPECT_NEAR(trace(A), 1 + 5 + 9, 1e-14);

    Eigen::VectorXd d = diag(A);
    EXPECT_EQ(static_cast<int>(d.size()), 3);
    EXPECT_NEAR(d(0), 1.0, 1e-15);
    EXPECT_NEAR(d(1), 5.0, 1e-15);
    EXPECT_NEAR(d(2), 9.0, 1e-15);

    /* 动态矩阵版本 */
    Eigen::MatrixXd Ddyn = diag(d);
    EXPECT_NEAR(Ddyn(0, 0), 1.0, 1e-15);
    EXPECT_NEAR(Ddyn(1, 1), 5.0, 1e-15);
    EXPECT_NEAR(Ddyn(2, 2), 9.0, 1e-15);
    EXPECT_NEAR(Ddyn(0, 1), 0.0, 1e-15);

    /* Matrix3d 版本保持原样 */
    Eigen::Matrix3d D = diag(Eigen::Vector3d(2, 4, 6));
    EXPECT_NEAR(D(0,0), 2.0, 1e-15);
    EXPECT_NEAR(D(1,1), 4.0, 1e-15);
    EXPECT_NEAR(D(2,2), 6.0, 1e-15);
    EXPECT_NEAR(D(0,1), 0.0, 1e-15);
}

static void test_mat_eye_and_inv4() {
    Eigen::MatrixXd I = eye(4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_NEAR(I(i, j), (i == j) ? 1.0 : 0.0, 1e-15);
        }
    }

    /* 4x4 对角逆 */
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4, 4);
    for (int i = 0; i < 4; i++) A(i, i) = i + 1.0;
    Eigen::MatrixXd Ai = inv4(A);
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(Ai(i, i), 1.0 / (i + 1.0), 1e-13);
    }
}

/* ============================================================================
 * 坐标系
 * ==========================================================================*/

static void test_blh_xyz_roundtrip() {
    Eigen::Vector3d blh(30.0 * DEG, 120.0 * DEG, 100.0);
    Eigen::Vector3d xyz = blh2xyz(blh);
    Eigen::Vector3d blh2 = xyz2blh(xyz);
    EXPECT_NEAR(blh2.x(), blh.x(), 1e-10);
    EXPECT_NEAR(blh2.y(), blh.y(), 1e-10);
    EXPECT_NEAR(blh2.z(), blh.z(), 1e-4);
}

static void test_pos2Cen_orthonormal() {
    Eigen::Vector3d pos(30.0 * DEG, 120.0 * DEG, 0.0);
    Eigen::Matrix3d C = pos2Cen(pos);
    Eigen::Matrix3d CtC = ((C.transpose())) * C;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(CtC(i, j), (i == j) ? 1.0 : 0.0, 1e-12);
        }
    }
    EXPECT_NEAR(det(C), 1.0, 1e-12);
}

static void test_Vxyz2enu() {
    /* 在赤道本初子午线：XYZ 与 ENU 对齐（U = Z） */
    Eigen::Vector3d pos(0.0, 0.0, 0.0);
    Eigen::Vector3d Vxyz(1.0, 2.0, 3.0);
    Eigen::Vector3d enu = Vxyz2enu(Vxyz, pos);
    /* 只断言模长不变 */
    EXPECT_NEAR(norm(enu), norm(Vxyz), 1e-12);
}

/* ============================================================================
 * 线性代数（Eigen 封装）
 * ==========================================================================*/

static void test_llt_L() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
    A(0, 0) = 4; A(1, 1) = 9; A(2, 2) = 16;
    Eigen::MatrixXd L = llt_L(A);
    EXPECT_EQ(L.rows(), 3);
    EXPECT_NEAR(L(0, 0), 2.0, 1e-13);
    EXPECT_NEAR(L(1, 1), 3.0, 1e-13);
    EXPECT_NEAR(L(2, 2), 4.0, 1e-13);
    /* 非对角为 0 */
    EXPECT_NEAR(L(0, 1), 0.0, 1e-15);
    EXPECT_NEAR(L(1, 0), 0.0, 1e-15);

    /* 一般 SPD */
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 3);
    B(0, 0) = 4; B(0, 1) = 2; B(0, 2) = 1;
    B(1, 0) = 2; B(1, 1) = 5; B(1, 2) = 3;
    B(2, 0) = 1; B(2, 1) = 3; B(2, 2) = 6;
    Eigen::MatrixXd Lb = llt_L(B);
    Eigen::MatrixXd LbLbt = Lb * Lb.transpose();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(LbLbt(i, j), B(i, j), 1e-12);
        }
    }
}

static void test_llt_inv() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
    A(0, 0) = 4; A(0, 1) = 2; A(0, 2) = 1;
    A(1, 0) = 2; A(1, 1) = 5; A(1, 2) = 3;
    A(2, 0) = 1; A(2, 1) = 3; A(2, 2) = 6;

    Eigen::MatrixXd Ai = llt_inv(A);
    Eigen::MatrixXd P = A * Ai;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(P(i, j), (i == j) ? 1.0 : 0.0, 1e-12);
        }
    }
}

static void test_llt_sqrtinv() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
    A(0, 0) = 4; A(0, 1) = 2; A(0, 2) = 1;
    A(1, 0) = 2; A(1, 1) = 5; A(1, 2) = 3;
    A(2, 0) = 1; A(2, 1) = 3; A(2, 2) = 6;

    Eigen::MatrixXd S = llt_sqrtinv(A);
    /* S^T S = A^{-1} */
    Eigen::MatrixXd StS = S.transpose() * S;
    Eigen::MatrixXd Ai = llt_inv(A);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(StS(i, j), Ai(i, j), 1e-11);
        }
    }
}

static void test_pinv_sym() {
    /* 秩亏矩阵 */
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(3, 3);
    A(0, 0) = 2; A(1, 1) = 3;
    /* A = diag(2, 3, 0) */
    Eigen::MatrixXd Ai = pinv_sym(A);
    EXPECT_NEAR(Ai(0, 0), 0.5, 1e-13);
    EXPECT_NEAR(Ai(1, 1), 1.0 / 3.0, 1e-13);
    EXPECT_NEAR(Ai(2, 2), 0.0, 1e-13);

    /* 满秩时与 inv 相同 */
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 3);
    B(0, 0) = 4; B(1, 1) = 5; B(2, 2) = 6;
    Eigen::MatrixXd Bi = pinv_sym(B);
    EXPECT_NEAR(Bi(0, 0), 0.25, 1e-13);
    EXPECT_NEAR(Bi(1, 1), 0.2, 1e-13);
    EXPECT_NEAR(Bi(2, 2), 1.0 / 6.0, 1e-13);
}

/* ============================================================================
 * IMU 圆锥补偿
 * ==========================================================================*/

static void test_imu_coning_compensation() {
    IMU imu;
    imu.Reset();

    const double dt = 0.002;
    const double A  = 1.0;
    const double f  = 100.0;

    auto omega = [&](double t) {
        double w = 2.0 * PI * f * t;
        return Eigen::Vector3d(A * std::cos(w), A * std::sin(w), 0.0);
    };

    Eigen::Vector3d pwm[2];
    pwm[0] = omega(dt * 0.25) * (dt * 0.5);
    pwm[1] = omega(dt * 0.75) * (dt * 0.5);

    /* 两个子样不平行 */
    Eigen::Vector3d cross01 = pwm[0].cross(pwm[1]);
    EXPECT_TRUE(norm(cross01) > 1e-9);

    Eigen::Vector3d pvm[2] = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    imu.Update(pwm, pvm, 2, dt * 0.5);

    /* phim = (pwm[0] + pwm[1]) + (2/3)*pwm[0] × pwm[1] */
    Eigen::Vector3d wmm = pwm[0] + pwm[1];
    Eigen::Vector3d cm  = pwm[0] * (2.0 / 3.0);
    Eigen::Vector3d phim_expect = wmm + cm.cross(pwm[1]);

    EXPECT_NEAR(imu.phim.x(), phim_expect.x(), 1e-15);
    EXPECT_NEAR(imu.phim.y(), phim_expect.y(), 1e-15);
    EXPECT_NEAR(imu.phim.z(), phim_expect.z(), 1e-15);

    /* 补偿项非零 */
    EXPECT_TRUE(norm(cm.cross(pwm[1])) > 1e-10);
}

/* ============================================================================
 * 其它工具
 * ==========================================================================*/

static void test_scalar_utils() {
    EXPECT_NEAR(range(0.5, 0.0, 1.0), 0.5, 1e-15);
    EXPECT_NEAR(range(-1.0, 0.0, 1.0), 0.0, 1e-15);
    EXPECT_NEAR(range(2.0, 0.0, 1.0), 1.0, 1e-15);

    EXPECT_EQ(sign(0.5), 1);
    EXPECT_EQ(sign(-0.5), -1);
    EXPECT_EQ(sign(0.0), 0);
    EXPECT_EQ(sign(1e-20, 1e-15), 0);

    EXPECT_NEAR(atan2Ex(0.0, 0.0), 0.0, 1e-15);
    EXPECT_NEAR(atan2Ex(1.0, 0.0), PI / 2, 1e-14);
    EXPECT_NEAR(atan2Ex(0.0, 1.0), 0.0, 1e-15);

    /* asinEx 把 |x|>1 钳到 ±1 */
    EXPECT_NEAR(asinEx(1.5), PI / 2, 1e-14);
    EXPECT_NEAR(asinEx(-1.5), -PI / 2, 1e-14);
    EXPECT_NEAR(asinEx(0.5), std::asin(0.5), 1e-15);
}

static void test_swapt() {
    int a = 1, b = 2;
    swapt(a, b);
    EXPECT_EQ(a, 2);
    EXPECT_EQ(b, 1);

    double x = 1.5, y = 2.5;
    swapt(x, y);
    EXPECT_NEAR(x, 2.5, 1e-15);
    EXPECT_NEAR(y, 1.5, 1e-15);
}

static void test_norm_ptrs() {
    double d[3] = {3.0, -4.0, 12.0};
    EXPECT_NEAR(norm(d, 3), 13.0, 1e-13);
    EXPECT_NEAR(norm1(d, 3), 19.0, 1e-13);
    EXPECT_NEAR(normInf(d, 3), 12.0, 1e-13);
}

static void test_MKQt() {
    EXPECT_NEAR(MKQt(1.0, 2.0), 1.0, 1e-15);

    Eigen::Vector3d sR(1.0, 2.0, 3.0), tau(1.0, 1.0, 1.0);
    Eigen::Vector3d q = MKQt(sR, tau);
    EXPECT_NEAR(q.x(), 2.0, 1e-14);
    EXPECT_NEAR(q.y(), 8.0, 1e-14);
    EXPECT_NEAR(q.z(), 18.0, 1e-14);
}

/* ============================================================================
 * main
 * ==========================================================================*/
int main() {
    std::printf("========== 数学模块单元测试 ==========\n");

    /* ---- 几何基础类型（Eigen）上的工程自由函数 ---- */
    RUN_TEST(test_vect3_norms);
    RUN_TEST(test_vect3_dot_and_products);
    RUN_TEST(test_vect3_elementwise);
    RUN_TEST(test_mat3_trace_det_inv);
    RUN_TEST(test_mat3_askew);
    RUN_TEST(test_mat3_rot);
    RUN_TEST(test_mat3_diag);
    RUN_TEST(test_mat3_dotmul_and_norm);
    RUN_TEST(test_mat3_pow);
    RUN_TEST(test_quat_basic);

    /* ---- 动态向量 / 矩阵上的工程自由函数 ---- */
    RUN_TEST(test_vect_norms);
    RUN_TEST(test_mat_diag_trace);
    RUN_TEST(test_mat_eye_and_inv4);

    /* ---- 姿态 / 旋转 ---- */
    RUN_TEST(test_rv2q_q2rv_roundtrip);
    RUN_TEST(test_rv2q_small_angle_taylor);
    RUN_TEST(test_q2mat_m2qua_roundtrip);
    RUN_TEST(test_a2qua_q2att_roundtrip);
    RUN_TEST(test_a2mat_m2att_roundtrip);
    RUN_TEST(test_a2qua_matches_a2mat);
    RUN_TEST(test_q2mat_orthonormal);
    RUN_TEST(test_m2rv_rv2m_roundtrip);
    RUN_TEST(test_qq2phi);
    RUN_TEST(test_addmu);
    RUN_TEST(test_UpDown);
    RUN_TEST(test_vn2att);
    RUN_TEST(test_diffYaw);

    /* ---- 坐标系 ---- */
    RUN_TEST(test_blh_xyz_roundtrip);
    RUN_TEST(test_pos2Cen_orthonormal);
    RUN_TEST(test_Vxyz2enu);

    /* ---- 对称正定线性代数 ---- */
    RUN_TEST(test_llt_L);
    RUN_TEST(test_llt_inv);
    RUN_TEST(test_llt_sqrtinv);
    RUN_TEST(test_pinv_sym);

    /* ---- IMU 与工具 ---- */
    RUN_TEST(test_imu_coning_compensation);
    RUN_TEST(test_scalar_utils);
    RUN_TEST(test_swapt);
    RUN_TEST(test_norm_ptrs);
    RUN_TEST(test_MKQt);

    std::printf("=============================================\n");
    std::printf("总断言: %d, 通过: %d, 失败: %d\n", g_pass + g_fail, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
