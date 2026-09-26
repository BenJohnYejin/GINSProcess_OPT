/**
 * @file marginalization_test.cpp
 * @brief 边缘化模块单元测试
 */
#include "alg_lib.h"

#include <cmath>
#include <cstdio>
#include <random>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do {                                       \
    if (!(cond)) { std::printf("[FAIL] %s\n", msg); ++g_fail; }     \
    else         { std::printf("[ OK ] %s\n", msg); ++g_pass; }     \
} while (0)

#define CHECK_NEAR(a, b, tol, msg) do {                             \
    double _a = (a), _b = (b);                                      \
    if (std::fabs(_a - _b) > (tol)) {                               \
        std::printf("[FAIL] %s: %.3e vs %.3e (diff=%.3e)\n",        \
                    msg, _a, _b, std::fabs(_a - _b));               \
        ++g_fail;                                                   \
    } else { std::printf("[ OK ] %s\n", msg); ++g_pass; }           \
} while (0)

/* ============================================================
 * T1：accumulate_normal_equations 基本正确性
 *   随机 J (p×n), r (p×1)，检查 H = JᵀJ, b = Jᵀr
 * ============================================================ */
static void test_accumulate() {
    std::printf("\n---- T1: accumulate_normal_equations ----\n");
    std::mt19937 rng(20240918u);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int trial = 0; trial < 5; ++trial) {
        const int p = 8, q = 5;
        Eigen::MatrixXd J(p, q);
        Eigen::VectorXd r(p);
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < q; ++j) J(i, j) = dist(rng);
            r(i) = dist(rng);
        }
        Eigen::MatrixXd H_ref = J.transpose() * J;
        Eigen::VectorXd b_ref = J.transpose() * r;

        mat H;
        vect b;
        accumulate_normal_equations(mat(J), vect(r), H, b);

        double dH = (H.toEigen() - H_ref).cwiseAbs().maxCoeff();
        double db = (b.toEigen() - b_ref).cwiseAbs().maxCoeff();
        CHECK_NEAR(dH, 0.0, 1e-12, "H = JᵀJ");
        CHECK_NEAR(db, 0.0, 1e-12, "b = Jᵀr");

        /* 二次累积 = 两倍 */
        mat H2 = H;
        vect b2 = b;
        accumulate_normal_equations(mat(J), vect(r), H2, b2);
        CHECK_NEAR((H2.toEigen() - 2.0 * H_ref).cwiseAbs().maxCoeff(), 0.0,
                   1e-12, "累积两次 = 2H");
    }
}

/* ============================================================
 * T2：Schur 补等价性
 *   边缘化前 m 维后，保留变量的最优解应与原始最小二乘一致
 *
 *   原问题：min ||J δx - r||²
 *   法方程：H δx = b，其中 H = JᵀJ, b = Jᵀr
 *   解：     δx_full = H⁻¹ b
 *
 *   边缘化后：H* δx_r = b*
 *   期望：   δx_r == δx_full.tail(r)
 * ============================================================ */
static void test_schur_equivalence() {
    std::printf("\n---- T2: Schur 补等价性 ----\n");
    std::mt19937 rng(20240918u);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int trial = 0; trial < 10; ++trial) {
        const int p = 20;   /* 残差个数 */
        const int q = 10;   /* 变量维数 */
        const int m = 4;    /* 边缘化前 m 维 */

        Eigen::MatrixXd J(p, q);
        Eigen::VectorXd r(p);
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < q; ++j) J(i, j) = dist(rng);
            r(i) = dist(rng);
        }

        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::VectorXd b = J.transpose() * r;

        /* --- 直接求解 --- */
        Eigen::VectorXd dx_full = H.ldlt().solve(b);
        Eigen::VectorXd dx_full_r = dx_full.tail(q - m);

        /* --- Schur 补 --- */
        mat J_out;
        vect r_out;
        const bool ok = schur_complement(mat(H), vect(b), m, J_out, r_out);
        CHECK(ok, "schur_complement 返回成功");

        /* --- 从 (J_out, r_out) 恢复法方程 --- */
        Eigen::MatrixXd J_out_e = J_out.toEigen();
        Eigen::VectorXd r_out_e = r_out.toEigen();
        Eigen::MatrixXd H_star = J_out_e.transpose() * J_out_e;
        Eigen::VectorXd b_star = J_out_e.transpose() * r_out_e;

        CHECK_NEAR(J_out.row, q - m, 0, "J_out 行数 = 保留维数");
        CHECK_NEAR(J_out.clm, q - m, 0, "J_out 列数 = 保留维数");

        /* --- 边缘化解 --- */
        Eigen::VectorXd dx_r = H_star.ldlt().solve(b_star);

        double err = (dx_r - dx_full_r).cwiseAbs().maxCoeff();
        CHECK(err < 1e-8, "边缘化解 = 完整解的后 r 维");
    }
}

/* ============================================================
 * T3：H_star 对称 + 半正定
 * ============================================================ */
static void test_psd() {
    std::printf("\n---- T3: H* 对称 + PSD ----\n");
    std::mt19937 rng(20240918u);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int trial = 0; trial < 5; ++trial) {
        const int p = 30, q = 8, m = 3;

        Eigen::MatrixXd J(p, q);
        Eigen::VectorXd r(p);
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < q; ++j) J(i, j) = dist(rng);
            r(i) = dist(rng);
        }
        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::VectorXd b = J.transpose() * r;

        mat J_out;
        vect r_out;
        schur_complement(mat(H), vect(b), m, J_out, r_out);

        Eigen::MatrixXd H_star = J_out.toEigen().transpose() * J_out.toEigen();

        /* 对称 */
        double asym = (H_star - H_star.transpose()).cwiseAbs().maxCoeff();
        CHECK_NEAR(asym, 0.0, 1e-12, "H* 对称");

        /* 半正定 */
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_star);
        CHECK(es.info() == Eigen::Success, "特征分解成功");
        CHECK(es.eigenvalues().minCoeff() > -1e-10, "H* 最小特征值 ≥ -1e-10");
    }
}

/* ============================================================
 * T4：边界条件
 *   m <= 0 或 m >= n → 返回 false
 *   m = 1 或 m = n-1 → 返回 true
 * ============================================================ */
static void test_boundary() {
    std::printf("\n---- T4: 边界条件 ----\n");
    std::mt19937 rng(20240918u);
    std::normal_distribution<double> dist(0.0, 1.0);

    const int n = 6;
    Eigen::MatrixXd J(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) J(i, j) = dist(rng);
    Eigen::MatrixXd H = J.transpose() * J + Eigen::MatrixXd::Identity(n, n);
    Eigen::VectorXd b = Eigen::VectorXd::Random(n);

    mat J_out;
    vect r_out;

    CHECK(!schur_complement(mat(H), vect(b),  0, J_out, r_out), "m=0 → false");
    CHECK(!schur_complement(mat(H), vect(b), -1, J_out, r_out), "m=-1 → false");
    CHECK(!schur_complement(mat(H), vect(b),  n, J_out, r_out), "m=n → false");
    CHECK( schur_complement(mat(H), vect(b),  1, J_out, r_out), "m=1 → true");
    CHECK( schur_complement(mat(H), vect(b),n-1, J_out, r_out), "m=n-1 → true");
    CHECK_NEAR(J_out.row, 1, 0, "m=n-1 时 J_out 为 1×1");
}

/* ============================================================
 * T5：跨窗口先验传递
 *   两次边缘化：先把旧状态消去得到先验，再加入新残差、再消去，
 *   总问题与"一次性把旧状态全消去"应该一致
 * ============================================================ */
static void test_schur_composition() {
    std::printf("\n---- T5a: Schur 补结合律 ----\n");
    std::mt19937 rng(20240918u);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int trial = 0; trial < 10; ++trial) {
        const int p  = 20;
        const int q  = 8;
        const int m1 = 2;
        const int m2 = 3;

        Eigen::MatrixXd J(p, q);
        Eigen::VectorXd r(p);
        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < q; ++j) J(i, j) = dist(rng);
            r(i) = dist(rng);
        }
        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::VectorXd b = J.transpose() * r;

        /* --- 一次性消 m1+m2 维 --- */
        mat  Jd;
        vect rd;
        CHECK(schur_complement(mat(H), vect(b), m1 + m2, Jd, rd),
              "一次性 schur 成功");

        /* --- 先消 m1 维 --- */
        mat  J1;
        vect r1;
        CHECK(schur_complement(mat(H), vect(b), m1, J1, r1),
              "第一次 schur 成功");

        Eigen::MatrixXd H1 = J1.toEigen().transpose() * J1.toEigen();
        Eigen::VectorXd b1 = J1.toEigen().transpose() * r1.toEigen();

        /* --- 再消 m2 维 --- */
        mat  J2;
        vect r2;
        CHECK(schur_complement(mat(H1), vect(b1), m2, J2, r2),
              "第二次 schur 成功");

        /* --- 比较最终等价的 (H*, b*) --- */
        Eigen::MatrixXd Hd = Jd.toEigen().transpose() * Jd.toEigen();
        Eigen::VectorXd bd = Jd.toEigen().transpose() * rd.toEigen();
        Eigen::MatrixXd H2 = J2.toEigen().transpose() * J2.toEigen();
        Eigen::VectorXd b2 = J2.toEigen().transpose() * r2.toEigen();

        CHECK_NEAR(Hd.rows(), H2.rows(), 0, "两次边缘化后维数一致");
        CHECK_NEAR((Hd - H2).cwiseAbs().maxCoeff(), 0.0, 1e-8,
                   "Schur 补结合律（H*）");
        CHECK_NEAR((bd - b2).cwiseAbs().maxCoeff(), 0.0, 1e-8,
                   "Schur 补结合律（b*）");
    }
}
static void test_cross_window_equivalence() {
    std::printf("\n---- T5b: 跨窗口等价性 ----\n");
    std::mt19937 rng(20240918u);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int trial = 0; trial < 10; ++trial) {
        const int p1 = 15, p2 = 15;
        const int q  = 10;
        const int m1 = 4, m2 = 3;

        /* J1：作用在全部 q 维上 */
        Eigen::MatrixXd J1(p1, q);
        Eigen::VectorXd r1(p1);
        for (int i = 0; i < p1; ++i) {
            for (int j = 0; j < q; ++j) J1(i, j) = dist(rng);
            r1(i) = dist(rng);
        }

        /* J2：只作用在后 q-m1 维上（前 m1 列为 0） */
        Eigen::MatrixXd J2 = Eigen::MatrixXd::Zero(p2, q);
        Eigen::VectorXd r2(p2);
        for (int i = 0; i < p2; ++i) {
            for (int j = m1; j < q; ++j) J2(i, j) = dist(rng);
            r2(i) = dist(rng);
        }

        /* --- 路径 A：两阶段 --- */
        mat  JA1;
        vect rA1;
        schur_complement(mat(J1.transpose() * J1),
                         vect(J1.transpose() * r1), m1, JA1, rA1);
        Eigen::MatrixXd HA1 = JA1.toEigen().transpose() * JA1.toEigen();
        Eigen::VectorXd bA1 = JA1.toEigen().transpose() * rA1.toEigen();

        Eigen::MatrixXd J2t = J2.rightCols(q - m1);
        Eigen::MatrixXd HA_total = HA1 + J2t.transpose() * J2t;
        Eigen::VectorXd bA_total = bA1 + J2t.transpose() * r2;

        mat  JA2;
        vect rA2;
        schur_complement(mat(HA_total), vect(bA_total), m2, JA2, rA2);

        /* --- 路径 B：一次性 --- */
        Eigen::MatrixXd J_all(p1 + p2, q);
        Eigen::VectorXd r_all(p1 + p2);
        J_all << J1, J2;
        r_all << r1, r2;

        mat  JB;
        vect rB;
        schur_complement(mat(J_all.transpose() * J_all),
                         vect(J_all.transpose() * r_all),
                         m1 + m2, JB, rB);

        /* --- 比较 --- */
        Eigen::MatrixXd HA = JA2.toEigen().transpose() * JA2.toEigen();
        Eigen::VectorXd bA = JA2.toEigen().transpose() * rA2.toEigen();
        Eigen::MatrixXd HB = JB.toEigen().transpose() * JB.toEigen();
        Eigen::VectorXd bB = JB.toEigen().transpose() * rB.toEigen();

        CHECK_NEAR((HA - HB).cwiseAbs().maxCoeff(), 0.0, 1e-8,
                   "跨窗口 H* 一致");
        CHECK_NEAR((bA - bB).cwiseAbs().maxCoeff(), 0.0, 1e-8,
                   "跨窗口 b* 一致");
    }
}
/* ============================================================
 * T6：退化 H_mm（含负特征值）的鲁棒性
 * ============================================================ */
static void test_degenerate_Hmm() {
    std::printf("\n---- T6: 退化 H_mm ----\n");

    /* H_mm 奇异：令前两维完全相同 */
    const int n = 5, m = 2, r = 3;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
    /* 前两行/列完全相同 → H_mm 秩 1，含 0 特征值 */
    H(0, 1) = H(1, 0) = 1.0;
    H(1, 1) = H(0, 0);

    Eigen::VectorXd b = Eigen::VectorXd::Ones(n);
    mat J_out; vect r_out;
    bool ok = schur_complement(mat(H), vect(b), m, J_out, r_out);
    CHECK(ok, "退化 H_mm 仍能返回成功");

    if (ok) {
        Eigen::MatrixXd HA = J_out.toEigen().transpose() * J_out.toEigen();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(HA);
        CHECK(es.eigenvalues().minCoeff() > -1e-10, "退化情形下 H* 仍 PSD");
    }
}

/* ============================================================ */
int main() {
    std::printf("========== Marginalization 单元测试 ==========\n");

    test_accumulate();
    test_schur_equivalence();
    test_psd();
    test_boundary();
    test_schur_composition();          // ← 新
    test_cross_window_equivalence();   // ← 新
    test_degenerate_Hmm();

    std::printf("\n=============================================\n");
    std::printf("通过 %d, 失败 %d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

