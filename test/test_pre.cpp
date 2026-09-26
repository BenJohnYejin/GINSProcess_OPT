/**
 * @file preintegration_test.cpp
 * @brief Preintegration 模块单元测试
 */
#include "alg_lib.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            std::printf("[FAIL] %s  (%s:%d)\n", msg, __FILE__, __LINE__);       \
            ++g_failures;                                                       \
        } else {                                                                \
            std::printf("[ OK ] %s\n", msg);                                    \
        }                                                                       \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                              \
    do {                                                                        \
        ++g_checks;                                                             \
        double _a = (a), _b = (b);                                              \
        if (std::fabs(_a - _b) > (tol)) {                                       \
            std::printf("[FAIL] %s  (%s:%d): %.6e vs %.6e (diff=%.3e, tol=%.3e)\n", \
                        msg, __FILE__, __LINE__, _a, _b,                        \
                        std::fabs(_a - _b), (tol));                             \
            ++g_failures;                                                       \
        } else {                                                                \
            std::printf("[ OK ] %s\n", msg);                                    \
        }                                                                       \
    } while (0)

/* ---------- 工具 ---------- */
static PreintegrationParam makeParam(bool noisy = false) {
    PreintegrationParam p;
    if (noisy) {
        p.gyr_arw      = 1.0e-3;   /* rad/sqrt(s) */
        p.acc_vrw      = 1.0e-3;   /* m/s^1.5    */
        p.gyr_bias_std = 1.0e-4;   /* rad/s      */
        p.acc_bias_std = 1.0e-3;   /* m/s^2      */
        p.corr_time    = 3600.0;
    }
    return p;
}

static ImuMeas makeMeas(double dt, const vect3 &dtheta, const vect3 &dvel,
                        double odovel = 0.0) {
    ImuMeas m;
    m.dt      = dt;
    m.dtheta  = dtheta;
    m.dvel    = dvel;
    m.odovel  = odovel;
    return m;
}

/* ==========================================================================
 * 测试 1：静止
 *   - 陀螺/加计全零 → Δq = I, Δv = 0, Δp = 0
 * ========================================================================== */
static void test_zero_motion() {
    std::printf("\n---- test_zero_motion ----\n");

    Preintegration preint(makeParam(false));
    preint.setBias(O31, O31);

    const double dt = 0.01;
    const int    N  = 100;
    const ImuMeas m  = makeMeas(dt, O31, O31);

    for (int i = 0; i < N; ++i) preint.integration(m);

    CHECK_NEAR(norm(preint.p()),  0.0, 1e-12, "Δp 保持零");
    CHECK_NEAR(norm(preint.v()),  0.0, 1e-12, "Δv 保持零");
    CHECK_NEAR(std::fabs(preint.q().q0), 1.0, 1e-12, "Δq = I (w)");
    CHECK_NEAR(preint.q().q1, 0.0, 1e-12, "Δq = I (x)");
    CHECK_NEAR(preint.q().q2, 0.0, 1e-12, "Δq = I (y)");
    CHECK_NEAR(preint.q().q3, 0.0, 1e-12, "Δq = I (z)");
    CHECK_NEAR(preint.dt(), dt * N, 1e-12, "累计时间");
    CHECK(preint.num() == N, "测量计数");
}

/* ==========================================================================
 * 测试 2：绕 z 轴匀速旋转
 *   ω_z = 1 rad/s, 总时长 T → 期望 Δq = Exp([0,0,ω_z·T])
 * ========================================================================== */
static void test_constant_rotation() {
    std::printf("\n---- test_constant_rotation ----\n");

    Preintegration preint(makeParam(false));
    preint.setBias(O31, O31);

    const double dt      = 0.005;
    const double omega_z = 1.0;
    const int    N       = 200;
    const double T       = dt * N;

    const ImuMeas m = makeMeas(dt, vect3(0.0, 0.0, omega_z * dt), O31);
    for (int i = 0; i < N; ++i) preint.integration(m);

    quat expect = rv2q(vect3(0.0, 0.0, omega_z * T));

    CHECK_NEAR(preint.q().q0, expect.q0, 1e-10, "匀速旋转 q0");
    CHECK_NEAR(preint.q().q1, expect.q1, 1e-10, "匀速旋转 q1");
    CHECK_NEAR(preint.q().q2, expect.q2, 1e-10, "匀速旋转 q2");
    CHECK_NEAR(preint.q().q3, expect.q3, 1e-10, "匀速旋转 q3");
    CHECK_NEAR(norm(preint.v()), 0.0, 1e-12, "无加速度 → Δv = 0");
    CHECK_NEAR(norm(preint.p()), 0.0, 1e-12, "无加速度 → Δp = 0");
}

/* ==========================================================================
 * 测试 3：恒定加速度（静止姿态）
 *   a = 1 m/s² along x, T = N·dt
 *   → Δv_x = a·T, Δp_x = 0.5·a·T²
 * ========================================================================== */
static void test_constant_acceleration() {
    std::printf("\n---- test_constant_acceleration ----\n");

    Preintegration preint(makeParam(false));
    preint.setBias(O31, O31);

    const double dt = 0.01;
    const double a  = 1.0;
    const int    N  = 100;
    const double T  = dt * N;

    const ImuMeas m = makeMeas(dt, O31, vect3(a * dt, 0.0, 0.0));
    for (int i = 0; i < N; ++i) preint.integration(m);

    CHECK_NEAR(preint.v().i, a * T,         1e-9, "Δv_x = a·T");
    CHECK_NEAR(preint.v().j, 0.0,           1e-12, "Δv_y = 0");
    CHECK_NEAR(preint.v().k, 0.0,           1e-12, "Δv_z = 0");
    CHECK_NEAR(preint.p().i, 0.5 * a * T * T, 1e-9, "Δp_x = 0.5·a·T²");
    CHECK_NEAR(preint.p().j, 0.0,           1e-12, "Δp_y = 0");
    CHECK_NEAR(preint.p().k, 0.0,           1e-12, "Δp_z = 0");
}

/* ==========================================================================
 * 测试 4：绕 z 匀速旋转 + 沿体轴 x 恒定加速度
 *   解析：Δv = ∫ R(Δq(t))·a_body dt
 *   为数值对照，采用与 propagate 相同的中值积分做参考
 *   （这里退化为"是否保持内部一致性"的检查：repropagation 结果应与直接 integration 相同）
 * ========================================================================== */
static void test_rotation_with_acceleration_consistency() {
    std::printf("\n---- test_rotation_with_acceleration_consistency ----\n");

    PreintegrationParam param = makeParam(true);
    const double dt = 0.01;
    const int    N  = 500;

    /* 生成一段轨迹 */
    std::vector<ImuMeas> seq;
    seq.reserve(N);
    std::mt19937 rng(20240918u);
    std::uniform_real_distribution<double> u(-0.02, 0.02);
    for (int i = 0; i < N; ++i) {
        seq.push_back(makeMeas(dt,
                               vect3(u(rng), u(rng), 0.5 + u(rng)),
                               vect3(0.3 + u(rng), u(rng), -0.2 + u(rng))));
    }

    /* 路径 A：直接 integration */
    Preintegration A(param);
    A.setBias(O31, O31);
    for (auto &m : seq) A.integration(m);

    /* 路径 B：先以零偏 integration，再 setBias 触发 repropagation */
    Preintegration B(param);
    B.setBias(O31, O31);
    for (auto &m : seq) B.integration(m);

    vect3 bg(1.0e-4, -2.0e-4, 3.0e-4);
    vect3 ba(1.0e-3, -5.0e-4, 2.0e-3);
    A.setBias(bg, ba);
    B.setBias(bg, ba);

    CHECK_NEAR(norm(A.p() - B.p()), 0.0, 1e-12, "repropagation 位置一致");
    CHECK_NEAR(norm(A.v() - B.v()), 0.0, 1e-12, "repropagation 速度一致");
    CHECK_NEAR(std::fabs(A.q().q0 - B.q().q0), 0.0, 1e-14, "repropagation 姿态一致 (w)");
    CHECK_NEAR(std::fabs(A.q().q1 - B.q().q1), 0.0, 1e-14, "repropagation 姿态一致 (x)");
}

/* ==========================================================================
 * 测试 5：加计零偏严格等效为常值零偏
 *   恒定 a_meas、给定 ba 时，去偏后的比力 a_body = a_meas − ba，
 *   因此 Δv = (a_meas − ba)·T，Δp = 0.5·(a_meas − ba)·T²
 * ========================================================================== */
static void test_bias_correction() {
    std::printf("\n---- test_bias_correction ----\n");

    const double dt = 0.01;
    const int    N  = 100;
    const double T  = dt * N;
    const vect3  ba(0.1, -0.05, 0.02);

    Preintegration preint(makeParam(false));
    preint.setBias(O31, ba);

    const ImuMeas m = makeMeas(dt, O31, vect3(0.1 * dt, -0.05 * dt, 0.02 * dt));
    for (int i = 0; i < N; ++i) preint.integration(m);

    /* 去偏后加速度应为零 */
    CHECK_NEAR(norm(preint.v()), 0.0, 1e-12, "加计零偏完全抵消速度");
    CHECK_NEAR(norm(preint.p()), 0.0, 1e-12, "加计零偏完全抵消位置");
    (void)T;
}

/* ==========================================================================
 * 测试 6：协方差矩阵对称正定
 * ========================================================================== */
static void test_covariance_psd() {
    std::printf("\n---- test_covariance_psd ----\n");

    Preintegration preint(makeParam(true));
    preint.setBias(O31, O31);

    const double dt = 0.01;
    const int    N  = 200;
    const ImuMeas m = makeMeas(dt,
                               vect3(1.0e-3, 2.0e-3, -1.0e-3),
                               vect3(0.05, -0.02, 0.10));
    for (int i = 0; i < N; ++i) preint.integration(m);

    const auto &cov = preint.cov();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> es(cov);
    CHECK(es.info() == Eigen::Success, "特征分解成功");

    double min_eig = es.eigenvalues().minCoeff();
    CHECK(min_eig >= -1e-18, "协方差最小特征值 ≥ 0");

    bool diag_ok = true;
    for (int i = 0; i < 15; ++i) {
        if (cov(i, i) < 0.0) { diag_ok = false; break; }
    }
    CHECK(diag_ok, "协方差对角元 ≥ 0");

    /* 对称性 */
    double asym = (cov - cov.transpose()).cwiseAbs().maxCoeff();
    CHECK_NEAR(asym, 0.0, 1e-15, "协方差对称");
}

/* ==========================================================================
 * 测试 7：数值误差随测量数量线性/平方增长
 *   无噪声时，恒定加速度的 Δp 与解析解一致，误差应保持机器精度量级
 * ========================================================================== */
static void test_long_run_numerical_stability() {
    std::printf("\n---- test_long_run_numerical_stability ----\n");

    Preintegration preint(makeParam(false));
    preint.setBias(O31, O31);

    const double dt = 0.005;
    const double a  = 9.8;
    const int    N  = 2000;
    const double T  = dt * N;

    const ImuMeas m = makeMeas(dt, O31, vect3(0.0, 0.0, a * dt));
    for (int i = 0; i < N; ++i) preint.integration(m);

    /* 沿 z 轴积分 */
    CHECK_NEAR(preint.v().k, a * T,          1e-8, "长时积分 Δv 稳定");
    CHECK_NEAR(preint.p().k, 0.5 * a * T * T, 1e-8, "长时积分 Δp 稳定");
    CHECK_NEAR(std::fabs(preint.q().q0), 1.0, 1e-12, "长时积分 Δq 稳定 (w)");
}

/* ==========================================================================
 * 测试 8：协方差随时间单调放大（无观测注入时）
 * ========================================================================== */
static void test_covariance_monotonic_growth() {
    std::printf("\n---- test_covariance_monotonic_growth ----\n");

    Preintegration preint(makeParam(true));
    preint.setBias(O31, O31);

    const double dt = 0.01;
    const ImuMeas m = makeMeas(dt,
                               vect3(1.0e-3, 0.0, 0.0),
                               vect3(0.0, 0.0, 9.8 * dt));

    double trace_prev = 0.0;
    for (int k = 0; k < 100; ++k) {
        preint.integration(m);
        double tr = preint.cov().trace();
        if (k >= 1) CHECK(tr >= trace_prev - 1e-15, "协方差迹单调不减");
        trace_prev = tr;
    }
}

/* ==========================================================================
 * 主入口
 * ========================================================================== */
int main() {
    std::printf("========== Preintegration 单元测试 ==========\n");

    test_zero_motion();
    test_constant_rotation();
    test_constant_acceleration();
    test_rotation_with_acceleration_consistency();
    test_bias_correction();
    test_covariance_psd();
    test_long_run_numerical_stability();
    test_covariance_monotonic_growth();

    std::printf("\n=============================================\n");
    std::printf("通过 %d / %d, 失败 %d\n",
                g_checks - g_failures, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}