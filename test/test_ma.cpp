/**
 * @file state_manifold_test.cpp
 * @brief StateManifold 模块单元测试
 */
#include "alg_lib.h"
#include <cstdio>
#include <random>


static int g_fail = 0, g_pass = 0;
#define CHECK(cond, msg) do {                                       \
    if (!(cond)) { std::printf("[FAIL] %s\n", msg); ++g_fail; }     \
    else         { std::printf("[ OK ] %s\n", msg); ++g_pass; }     \
} while (0)

/* ---- 工具函数 ---- */
static State makeRandomState(std::mt19937 &rng) {
    std::normal_distribution<double> n(0.0, 1.0);
    State s;
    s.p  = vect3(n(rng), n(rng), n(rng));
    s.q  = rv2q(vect3(n(rng), n(rng), n(rng)));  s.q.normlize(&s.q);
    s.v  = vect3(n(rng), n(rng), n(rng));
    s.bg = vect3(n(rng), n(rng), n(rng));
    s.ba = vect3(n(rng), n(rng), n(rng));
    s.sodo = n(rng);
    return s;
}

static DeltaN makeRandomDelta(std::mt19937 &rng, double scale = 1e-3) {
    DeltaN d;
    std::normal_distribution<double> n(0.0, 1.0);
    for (int i = 0; i < NUM_STATE; ++i) d(i) = scale * n(rng);
    return d;
}

/* ============================================================
 * T1：零扰动恒等性  x ⊞ 0 = x
 * ============================================================ */
static void test_zero_delta_identity() {
    std::printf("\n---- T1: x ⊞ 0 = x ----\n");
    std::mt19937 rng(20240918u);
    for (int k = 0; k < 20; ++k) {
        State x = makeRandomState(rng);
        DeltaN zero = DeltaN::Zero();
        State y = plus(x, zero);

        CHECK((y.p - x.p).toEigen().norm() < 1e-14, "zero-delta p");
        CHECK((y.v - x.v).toEigen().norm() < 1e-14, "zero-delta v");
        CHECK(std::fabs(y.q.q0 - x.q.q0) < 1e-14, "zero-delta q0");
        CHECK((q2rv((~x.q) * y.q)).toEigen().norm() < 1e-14, "zero-delta dphi");
    }
}

/* ============================================================
 * T2：往返一致性  (x ⊞ δ) ⊟ x = δ
 * ============================================================ */
static void test_roundtrip_plus_minus() {
    std::printf("\n---- T2: (x ⊞ δ) ⊟ x = δ ----\n");
    std::mt19937 rng(20240918u);
    for (int k = 0; k < 50; ++k) {
        State x = makeRandomState(rng);
        DeltaN d = makeRandomDelta(rng, 1e-2);

        State y = plus(x, d);
        DeltaN d_rec = minus(y, x);

        double err = (d_rec - d).cwiseAbs().maxCoeff();
        CHECK(err < 1e-10, "roundtrip plus→minus");
    }
}

/* ============================================================
 * T3：往返一致性  (x1 ⊟ x2) ⊞ x2 = x1
 * ============================================================ */
static void test_roundtrip_minus_plus() {
    std::printf("\n---- T3: (x1 ⊟ x2) ⊞ x2 = x1 ----\n");
    std::mt19937 rng(20240918u);
    for (int k = 0; k < 50; ++k) {
        State x1 = makeRandomState(rng);
        State x2 = makeRandomState(rng);

        DeltaN d = minus(x1, x2);
        State x1_rec = plus(x2, d);

        double dp_err = (x1_rec.p - x1.p).toEigen().norm();
        double dv_err = (x1_rec.v - x1.v).toEigen().norm();
        double dq_err = (q2rv((~x1.q) * x1_rec.q)).toEigen().norm();
        double dbg_err = (x1_rec.bg - x1.bg).toEigen().norm();
        double dba_err = (x1_rec.ba - x1.ba).toEigen().norm();

        CHECK(dp_err  < 1e-10, "roundtrip minus→plus p");
        CHECK(dv_err  < 1e-10, "roundtrip minus→plus v");
        CHECK(dq_err  < 1e-10, "roundtrip minus→plus q");
        CHECK(dbg_err < 1e-10, "roundtrip minus→plus bg");
        CHECK(dba_err < 1e-10, "roundtrip minus→plus ba");
    }
}

/* ============================================================
 * T4：纯旋转扰动应等价于 q ⊗ Exp(δφ)
 * ============================================================ */
static void test_rotation_perturbation() {
    std::printf("\n---- T4: 旋转扰动语义 ----\n");
    std::mt19937 rng(20240918u);
    std::normal_distribution<double> n(0.0, 1.0);

    for (int k = 0; k < 20; ++k) {
        State x;
        x.q = rv2q(vect3(n(rng), n(rng), n(rng)));
        x.q.normlize(&x.q);

        vect3 dphi(n(rng) * 1e-3, n(rng) * 1e-3, n(rng) * 1e-3);

        DeltaN d = DeltaN::Zero();
        d(3) = dphi.i; d(4) = dphi.j; d(5) = dphi.k;

        State y = plus(x, d);

        /* 期望：y.q == x.q ⊗ Exp(dphi) */
        quat q_expect = x.q * rv2q(dphi);
        q_expect.normlize(&q_expect);

        double dq = std::sqrt(std::pow(y.q.q0 - q_expect.q0, 2) +
                              std::pow(y.q.q1 - q_expect.q1, 2) +
                              std::pow(y.q.q2 - q_expect.q2, 2) +
                              std::pow(y.q.q3 - q_expect.q3, 2));
        CHECK(dq < 1e-14, "旋转扰动 = q ⊗ Exp(dphi)");
    }
}

/* ============================================================
 * T5：小角度极限  Log(Exp(δφ)) ≈ δφ
 * ============================================================ */
static void test_small_angle_limit() {
    std::printf("\n---- T5: 小角度 Log∘Exp ----\n");
    std::mt19937 rng(20240918u);
    std::normal_distribution<double> n(0.0, 1.0);

    for (double scale : {1e-3, 1e-5, 1e-8}) {
        State x;
        State y;
        vect3 dphi(scale * n(rng), scale * n(rng), scale * n(rng));
        y.q = rv2q(dphi);

        DeltaN d = minus(y, x);   /* x.q = I */
        vect3 dphi_back(d(3), d(4), d(5));

        double err = (dphi_back - dphi).toEigen().norm();
        CHECK(err < scale * 1e-6, "小角度近似精度");
    }
}

/* ============================================================
 * T6：与现有 posePlus / poseMinus 一致性
 * ============================================================ */
static void test_consistency_with_pose_ops() {
    std::printf("\n---- T6: 与 posePlus/poseMinus 一致 ----\n");
    std::mt19937 rng(20240918u);

    for (int k = 0; k < 20; ++k) {
        State x  = makeRandomState(rng);
        State x2 = makeRandomState(rng);
        DeltaN d = makeRandomDelta(rng, 1e-2);

        double pose_x [7], mix_x [9];
        double pose_x2[7], mix_x2[9];
        toData(x,  pose_x,  mix_x,  false);
        toData(x2, pose_x2, mix_x2, false);

        /* ---------- posePlus vs StateManifold::plus ---------- */
        double delta[6] = {d(0), d(1), d(2), d(3), d(4), d(5)};
        double pose_y[7];
        posePlus(pose_x, delta, pose_y);

        State y = plus(x, d);

        double dp_err = std::sqrt(std::pow(pose_y[0] - y.p.i, 2) +
                                  std::pow(pose_y[1] - y.p.j, 2) +
                                  std::pow(pose_y[2] - y.p.k, 2));
        /* 用四元数点积的绝对值比较，避免 q 与 -q 表示同一旋转时被误判 */
        double q_dot = std::fabs(pose_y[3] * y.q.q1 + pose_y[4] * y.q.q2 +
                                 pose_y[5] * y.q.q3 + pose_y[6] * y.q.q0);
        CHECK(dp_err < 1e-14, "plus ↔ posePlus 位置一致");
        CHECK(q_dot > 1.0 - 1e-12, "plus ↔ posePlus 姿态一致");

        /* ---------- poseMinus vs StateManifold::minus ----------
         * 关键：不要比较 q2rv 的输出（对 q0 符号敏感）。
         * 直接比较两边还原出的相对旋转四元数。 */
        double delta_out[6];
        poseMinus(pose_y, pose_x2, delta_out);

        /* poseMinus 内部 dphi = Log(qx2⁻¹ ⊗ qy)，
         * 我们把它还原成 quat 再和 minus 的等价量比较 */
        quat q_delta_pose = rv2q(vect3(delta_out[3], delta_out[4], delta_out[5]));

        DeltaN dm = minus(y, x2);
        quat q_delta_mine = rv2q(vect3(dm(3), dm(4), dm(5)));

        /* 两个四元数表示同一旋转 ⟺ |<q1, q2>| = 1 */
        double dq = std::fabs(q_delta_pose.q0 * q_delta_mine.q0 +
                              q_delta_pose.q1 * q_delta_mine.q1 +
                              q_delta_pose.q2 * q_delta_mine.q2 +
                              q_delta_pose.q3 * q_delta_mine.q3);
        CHECK(dq > 1.0 - 1e-12, "minus ↔ poseMinus 旋转一致（四元数点积）");

        /* 同时可以顺带检查相对旋转角度误差 */
        double theta_pose = 2.0 * std::acos(range(std::fabs(q_delta_pose.q0), 0.0, 1.0));
        double theta_mine = 2.0 * std::acos(range(std::fabs(q_delta_mine.q0), 0.0, 1.0));
        CHECK(std::fabs(theta_pose - theta_mine) < 1e-12,
              "minus ↔ poseMinus 旋转角一致");
    }
}

/* ============================================================
 * T7：16 维流形（含 sodo）
 * ============================================================ */
static void test_16dim_with_odometer() {
    std::printf("\n---- T7: 16 维流形（sodo） ----\n");
    std::mt19937 rng(20240918u);

    for (int k = 0; k < 20; ++k) {
        State x = makeRandomState(rng);

        DeltaNOdo d;
        std::normal_distribution<double> n(0.0, 1.0);
        for (int i = 0; i < NUM_STATE + 1; ++i) d(i) = 1e-3 * n(rng);

        State y = plus(x, d);
        DeltaNOdo d_rec = minusWithOdo(y, x);

        double err = (d_rec - d).cwiseAbs().maxCoeff();
        CHECK(err < 1e-10, "16 维往返一致性");
        CHECK(std::fabs((y.sodo - x.sodo) - d(15)) < 1e-14, "sodo 加性");
    }
}

/* ============================================================
 * T8：toData / fromData 往返
 * ============================================================ */
static void test_data_bridge() {
    std::printf("\n---- T8: toData/fromData 往返 ----\n");
    std::mt19937 rng(20240918u);

    for (bool with_odo : {false, true}) {
        for (int k = 0; k < 20; ++k) {
            State x = makeRandomState(rng);
            double pose[7], mix[10];
            toData(x, pose, mix, with_odo);
            State y = fromData(pose, mix, with_odo);

            double dp = (x.p - y.p).toEigen().norm();
            double dv = (x.v - y.v).toEigen().norm();
            double dq = (q2rv((~x.q) * y.q)).toEigen().norm();
            double dbg = (x.bg - y.bg).toEigen().norm();
            double dba = (x.ba - y.ba).toEigen().norm();
            double ds = with_odo ? std::fabs(x.sodo - y.sodo) : 0.0;
            CHECK(dp < 1e-14 && dv < 1e-14 && dq < 1e-14 &&
                dbg < 1e-14 && dba < 1e-14 &&
                (!with_odo || ds < 1e-14),
                with_odo ? "toData/fromData 含里程计" : "toData/fromData 无里程计");
        }
    }
}

/* ============================================================
 * T9：applyDelta 原地更新等价于 plus
 * ============================================================ */
static void test_applyDelta() {
    std::printf("\n---- T9: applyDelta 原地更新 ----\n");
    std::mt19937 rng(20240918u);

    for (int k = 0; k < 20; ++k) {
        State x1 = makeRandomState(rng);
        State x2 = x1;
        DeltaN d = makeRandomDelta(rng, 1e-2);

        State y = plus(x1, d);
        applyDelta(x2, d);

        double dp = (y.p - x2.p).toEigen().norm();
        double dv = (y.v - x2.v).toEigen().norm();
        double dq = (q2rv((~y.q) * x2.q)).toEigen().norm();
        CHECK(dp < 1e-14 && dv < 1e-14 && dq < 1e-14, "applyDelta == plus");
    }
}

/* ============================================================
 * T10：有限差分验证雅可比结构（用于 Ceres）
 *      验证 ∂(x ⊞ δ) / ∂δ 在 δ=0 处的结构
 * ============================================================ */
static void test_jacobian_structure() {
    std::printf("\n---- T10: 数值雅可比结构 ----\n");
    std::mt19937 rng(20240918u);
    std::normal_distribution<double> n(0.0, 1.0);

    State x;
    x.q = rv2q(vect3(0.3 * n(rng), -0.5 * n(rng), 0.8 * n(rng)));
    x.q.normlize(&x.q);

    const double h = 1e-7;

    /* 对 δp 的雅可比：应等于 I */
    for (int i = 0; i < 3; ++i) {
        DeltaN d = DeltaN::Zero();
        d(i) = h;
        State y = plus(x, d);
        vect3 dp = y.p - x.p;
        CHECK(std::fabs(dp.i / h - (i == 0 ? 1.0 : 0.0)) < 1e-9, "J_p = I");
    }

    /* 对 δbg 的雅可比：应等于 I */
    for (int i = 0; i < 3; ++i) {
        DeltaN d = DeltaN::Zero();
        d(9 + i) = h;
        State y = plus(x, d);
        vect3 dbg = y.bg - x.bg;
        CHECK(std::fabs(dbg.i / h - (i == 0 ? 1.0 : 0.0)) < 1e-9, "J_bg = I");
    }

    /* 对 δφ 的雅可比：应为 -R(q)（李代数左乘结构） */
    for (int i = 0; i < 3; ++i) {
        DeltaN d = DeltaN::Zero();
        d(3 + i) = h;
        State y = plus(x, d);
        vect3 dphi = q2rv((~x.q) * y.q);
        CHECK(std::fabs(dphi.i / h - (i == 0 ? 1.0 : 0.0)) < 1e-6, "J_phi (右扰动)");
    }
}

/* ============================================================ */
int main() {
    std::printf("========== StateManifold 单元测试 ==========\n");

    test_zero_delta_identity();
    test_roundtrip_plus_minus();
    test_roundtrip_minus_plus();
    test_rotation_perturbation();
    test_small_angle_limit();
    test_consistency_with_pose_ops();
    test_16dim_with_odometer();
    test_data_bridge();
    test_applyDelta();
    test_jacobian_structure();

    std::printf("\n=============================================\n");
    std::printf("通过 %d, 失败 %d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}