/* ============================================================================
 * test_marg.cpp —— 边缘化相关测试（合并后）
 *
 * 本文件由以下原文件合并而成（各自在独立 namespace 内，互不干扰）：
 *   - test_marg.cpp        accumulate_normal_equations / schur_complement
 *   - test_marg_prior.cpp  边缘化先验（多参数块 + 流形）的等价性验收
 *   - test_runner_marg.cpp 滑窗边缘化的端到端测试（合成轨迹跑真实 runner）
 * ==========================================================================*/
#include "alg_lib.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

/* 统一计数与判据（各 namespace 内另有同名的局部计数，宏在其作用域内解析到各自的那一份） */
#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (cond) { ++g_pass; }                                                \
        else {                                                                 \
            ++g_fail;                                                          \
            std::printf("[FAIL] %s  (%s:%d)\n", msg, __FILE__, __LINE__);     \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                             \
    do {                                                                       \
        const double _a = (a), _b = (b), _t = (tol);                           \
        if (std::fabs(_a - _b) <= _t) { ++g_pass; }                            \
        else {                                                                 \
            ++g_fail;                                                          \
            std::printf("[FAIL] %s  (%s:%d): %.6g vs %.6g (|d|=%.3g > %g)\n", \
                        msg, __FILE__, __LINE__, _a, _b,                        \
                        std::fabs(_a - _b), _t);                               \
        }                                                                      \
    } while (0)

/* ============================================================================
 * 以下为原 test/test_marg.cpp 的内容，已并入本文件
 * ==========================================================================*/
namespace marg_base {
    /**
     * @file marginalization_test.cpp
     * @brief 边缘化模块单元测试
     */


    static int g_pass = 0, g_fail = 0;



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

            Eigen::MatrixXd H;
            Eigen::VectorXd b;
            accumulate_normal_equations((J), (r), H, b);

            double dH = (H - H_ref).cwiseAbs().maxCoeff();
            double db = (b - b_ref).cwiseAbs().maxCoeff();
            CHECK_NEAR(dH, 0.0, 1e-12, "H = JᵀJ");
            CHECK_NEAR(db, 0.0, 1e-12, "b = Jᵀr");

            /* 二次累积 = 两倍 */
            Eigen::MatrixXd H2 = H;
            Eigen::VectorXd b2 = b;
            accumulate_normal_equations((J), (r), H2, b2);
            CHECK_NEAR((H2 - 2.0 * H_ref).cwiseAbs().maxCoeff(), 0.0,
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
            Eigen::MatrixXd J_out;
            Eigen::VectorXd r_out;
            const bool ok = schur_complement((H), (b), m, J_out, r_out);
            CHECK(ok, "schur_complement 返回成功");

            /* --- 从 (J_out, r_out) 恢复法方程 --- */
            Eigen::MatrixXd J_out_e = J_out;
            Eigen::VectorXd r_out_e = r_out;
            Eigen::MatrixXd H_star = J_out_e.transpose() * J_out_e;
            Eigen::VectorXd b_star = J_out_e.transpose() * r_out_e;

            CHECK_NEAR(J_out.rows(), q - m, 0, "J_out 行数 = 保留维数");
            CHECK_NEAR(J_out.cols(), q - m, 0, "J_out 列数 = 保留维数");

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

            Eigen::MatrixXd J_out;
            Eigen::VectorXd r_out;
            schur_complement((H), (b), m, J_out, r_out);

            Eigen::MatrixXd H_star = J_out.transpose() * J_out;

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

        Eigen::MatrixXd J_out;
        Eigen::VectorXd r_out;

        CHECK(!schur_complement((H), (b),  0, J_out, r_out), "m=0 → false");
        CHECK(!schur_complement((H), (b), -1, J_out, r_out), "m=-1 → false");
        CHECK(!schur_complement((H), (b),  n, J_out, r_out), "m=n → false");
        CHECK( schur_complement((H), (b),  1, J_out, r_out), "m=1 → true");
        CHECK( schur_complement((H), (b),n-1, J_out, r_out), "m=n-1 → true");
        CHECK_NEAR(J_out.rows(), 1, 0, "m=n-1 时 J_out 为 1×1");
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
            Eigen::MatrixXd  Jd;
            Eigen::VectorXd rd;
            CHECK(schur_complement((H), (b), m1 + m2, Jd, rd),
                  "一次性 schur 成功");

            /* --- 先消 m1 维 --- */
            Eigen::MatrixXd  J1;
            Eigen::VectorXd r1;
            CHECK(schur_complement((H), (b), m1, J1, r1),
                  "第一次 schur 成功");

            Eigen::MatrixXd H1 = J1.transpose() * J1;
            Eigen::VectorXd b1 = J1.transpose() * r1;

            /* --- 再消 m2 维 --- */
            Eigen::MatrixXd  J2;
            Eigen::VectorXd r2;
            CHECK(schur_complement((H1), (b1), m2, J2, r2),
                  "第二次 schur 成功");

            /* --- 比较最终等价的 (H*, b*) --- */
            Eigen::MatrixXd Hd = Jd.transpose() * Jd;
            Eigen::VectorXd bd = Jd.transpose() * rd;
            Eigen::MatrixXd H2 = J2.transpose() * J2;
            Eigen::VectorXd b2 = J2.transpose() * r2;

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
            Eigen::MatrixXd  JA1;
            Eigen::VectorXd rA1;
            schur_complement((J1.transpose() * J1),
                             (J1.transpose() * r1), m1, JA1, rA1);
            Eigen::MatrixXd HA1 = JA1.transpose() * JA1;
            Eigen::VectorXd bA1 = JA1.transpose() * rA1;

            Eigen::MatrixXd J2t = J2.rightCols(q - m1);
            Eigen::MatrixXd HA_total = HA1 + J2t.transpose() * J2t;
            Eigen::VectorXd bA_total = bA1 + J2t.transpose() * r2;

            Eigen::MatrixXd  JA2;
            Eigen::VectorXd rA2;
            schur_complement((HA_total), (bA_total), m2, JA2, rA2);

            /* --- 路径 B：一次性 --- */
            Eigen::MatrixXd J_all(p1 + p2, q);
            Eigen::VectorXd r_all(p1 + p2);
            J_all << J1, J2;
            r_all << r1, r2;

            Eigen::MatrixXd  JB;
            Eigen::VectorXd rB;
            schur_complement((J_all.transpose() * J_all),
                             (J_all.transpose() * r_all),
                             m1 + m2, JB, rB);

            /* --- 比较 --- */
            Eigen::MatrixXd HA = JA2.transpose() * JA2;
            Eigen::VectorXd bA = JA2.transpose() * rA2;
            Eigen::MatrixXd HB = JB.transpose() * JB;
            Eigen::VectorXd bB = JB.transpose() * rB;

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
        Eigen::MatrixXd J_out; Eigen::VectorXd r_out;
        bool ok = schur_complement((H), (b), m, J_out, r_out);
        CHECK(ok, "退化 H_mm 仍能返回成功");

        if (ok) {
            Eigen::MatrixXd HA = J_out.transpose() * J_out;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(HA);
            CHECK(es.eigenvalues().minCoeff() > -1e-10, "退化情形下 H* 仍 PSD");
        }
    }

    /* ============================================================ */
    int run() {
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
}  /* namespace marg_base */

/* ============================================================================
 * 以下为原 test/test_marg_prior.cpp 的内容，已并入本文件
 * ==========================================================================*/
namespace marg_prior {
    /* ============================================================================
     * test_marg_prior.cpp —— 边缘化先验的验收测试
     *
     * 原先这个文件（.review/stale/ 里的旧版）引用了一套已不存在的 GraphOptimizer API，
     * 既不在 CMakeLists 里、也编译不过。这里按**当前** API 重写，重点验证两件事：
     *
     *   T1  MarginalizationPriorFactor 在 Ceres 眼里就是"切空间雅可比 = J_out、残差 = r_out"。
     *       这等价于验证  ∂ρ/∂ambient = J_out · MinusJacobian(x)  这条链式法则是对的
     *       （Ceres 会用 tangent_jac = ambient_jac · PlusJacobian 反推，
     *        若 MinusJacobian·PlusJacobian ≠ I 或公式写错，这里就会失败）。
     *
     *   T2  边缘化的**语义**正确性（最关键的一条）：
     *       把 frame0 边缘化掉之后，"只剩保留帧的因子 + 先验"的最优解，
     *       必须与"完整问题"的最优解在保留变量上完全一致。
     *       同时确认先验确实起了作用（否则退化成丢帧）。
     *
     * 为什么 T2 能验出常见的两类错误：
     *   - 若把"不触及 frame0 的残差"也放进 Schur 补 → 信息被计两次 → 解偏（过度自信）；
     *   - 若先验符号取反（ρ = J·δ − r）→ 解被拉向反方向；
     *   两种情形下 x_red 都不等于 x_full。
     * ==========================================================================*/


    static int g_pass = 0;
    static int g_fail = 0;



    /* ============================================================================
     * 测试用因子
     * ==========================================================================*/

    /* 连接 (pose_i, mix_i) 与 (pose_j, mix_j) 的耦合因子（形如预积分残差）
     *   r0 = w·[(p_j − p_i) − R(q_i)·dp]
     *   r1 = w·[(q_i⁻¹ ⊗ q_j).vec() − dphi]
     *   r2 = w·[(v_j − v_i) − R(q_i)·dv]
     * 9 维，块：pose_i(7, PoseManifold) / mix_i(9) / pose_j(7, PoseManifold) / mix_j(9) */
    struct CoupledResidual {
        CoupledResidual(const Eigen::Vector3d &dp, const Eigen::Vector3d &dphi,
                        const Eigen::Vector3d &dv, double w)
            : dp_(dp), dphi_(dphi), dv_(dv), w_(w) {}

        template <typename T>
        bool operator()(const T *const pi, const T *const mi,
                        const T *const pj, const T *const mj, T *res) const {
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_i(pi);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_i(mi);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_j(pj);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_j(mj);
            Eigen::Quaternion<T> q_i(pi[6], pi[3], pi[4], pi[5]);   /* w 在 index 6 */
            Eigen::Quaternion<T> q_j(pj[6], pj[3], pj[4], pj[5]);
            q_i.normalize();
            q_j.normalize();
            const Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();

            Eigen::Map<Eigen::Matrix<T, 3, 1>> r0(res);
            Eigen::Map<Eigen::Matrix<T, 3, 1>> r1(res + 3);
            Eigen::Map<Eigen::Matrix<T, 3, 1>> r2(res + 6);
            const T w = T(w_);
            r0 = w * ((p_j - p_i) - R_i * dp_.cast<T>());
            r1 = w * ((q_i.conjugate() * q_j).vec() - dphi_.cast<T>());
            r2 = w * ((v_j - v_i) - R_i * dv_.cast<T>());
            return true;
        }

        Eigen::Vector3d dp_, dphi_, dv_;
        double w_;
    };

    /* 对单个 pose 的弱量测（同时起 gauge fix 作用）：6 维 */
    struct PoseMeas {
        PoseMeas(const Eigen::Vector3d &p_ref, const Eigen::Vector3d &qvec_ref, double w)
            : p_ref_(p_ref), qvec_ref_(qvec_ref), w_(w) {}

        template <typename T>
        bool operator()(const T *const pose, T *res) const {
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> p(pose);
            Eigen::Map<Eigen::Matrix<T, 3, 1>> r0(res);
            Eigen::Map<Eigen::Matrix<T, 3, 1>> r1(res + 3);
            r0 = T(w_) * (p - p_ref_.cast<T>());
            r1 = T(w_) * (Eigen::Matrix<T, 3, 1>(pose[3], pose[4], pose[5]) - qvec_ref_.cast<T>());
            return true;
        }
        Eigen::Vector3d p_ref_, qvec_ref_;
        double w_;
    };

    /* 对 mix 的弱量测：9 维 */
    struct MixMeas {
        MixMeas(const Eigen::VectorXd &ref, double w) : ref_(ref), w_(w) {}
        template <typename T>
        bool operator()(const T *const mix, T *res) const {
            for (int k = 0; k < 9; ++k) res[k] = T(w_) * (mix[k] - T(ref_(k)));
            return true;
        }
        Eigen::VectorXd ref_;
        double w_;
    };

    /* ============================================================================
     * 工具
     * ==========================================================================*/
    static void setPose(double *pose, const Eigen::Vector3d &p, const Eigen::Vector3d &qvec) {
        pose[0] = p.x(); pose[1] = p.y(); pose[2] = p.z();
        pose[3] = qvec.x(); pose[4] = qvec.y(); pose[5] = qvec.z();
        pose[6] = std::sqrt(std::max(0.0, 1.0 - qvec.squaredNorm()));
    }

    static void solveProblem(ceres::Problem &problem) {
        ceres::Solver::Options o;
        o.linear_solver_type       = ceres::DENSE_QR;
        o.max_num_iterations       = 200;
        o.function_tolerance       = 1e-15;
        o.gradient_tolerance       = 1e-15;
        o.parameter_tolerance      = 1e-15;
        o.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary s;
        ceres::Solve(o, &problem, &s);
    }

    /* 用显式列序求 J/r，并展成稠密 */
    static bool evalDense(ceres::Problem &problem,
                          const std::vector<double*> &blocks,
                          const std::vector<ceres::ResidualBlockId> &res_ids,
                          Eigen::MatrixXd &J, Eigen::VectorXd &r) {
        ceres::Problem::EvaluateOptions eo;
        eo.apply_loss_function = false;
        eo.parameter_blocks    = blocks;
        if (!res_ids.empty()) eo.residual_blocks = res_ids;

        double cost = 0.0;
        std::vector<double> residuals, gradient;
        ceres::CRSMatrix jac;
        if (!problem.Evaluate(eo, &cost, &residuals, &gradient, &jac)) return false;

        J = Eigen::MatrixXd::Zero(jac.num_rows, jac.num_cols);
        for (int row = 0; row < jac.num_rows; ++row) {
            for (int k = jac.rows[row]; k < jac.rows[row + 1]; ++k) {
                J(row, jac.cols[k]) = jac.values[k];
            }
        }
        r.resize(static_cast<int>(residuals.size()));
        for (size_t k = 0; k < residuals.size(); ++k) r(static_cast<int>(k)) = residuals[k];
        return true;
    }

    /* ============================================================================
     * T1：先验在 Ceres 眼里的切空间雅可比 == J_out，残差 == r_out
     * ==========================================================================*/
    static void test_prior_tangent_jacobian() {
        std::printf("\n---- T1: 先验的切空间雅可比 ----\n");

        /* 造一个 3 块（pose 7 + mix 9 + mix 9）的先验，J_out 是 15×24 的任意矩阵 */
        const int n_tan = 24;
        Eigen::MatrixXd J_out = Eigen::MatrixXd::Zero(15, n_tan);
        for (int i = 0; i < 15; ++i)
            for (int j = 0; j < n_tan; ++j)
                J_out(i, j) = 0.37 * std::sin(1.7 * i + 0.9 * j) + (i == j ? 1.3 : 0.0);
        Eigen::VectorXd r_out(15);
        for (int i = 0; i < 15; ++i) r_out(i) = 0.21 * std::cos(0.8 * i);

        std::vector<double> pose(7, 0.0), mix_a(9, 0.0), mix_b(9, 0.0);
        setPose(pose.data(), Eigen::Vector3d(1.0, -2.0, 0.5), Eigen::Vector3d(0.05, 0.02, -0.03));
        for (int k = 0; k < 9; ++k) { mix_a[k] = 0.1 * k; mix_b[k] = -0.05 * k; }

        std::vector<MargPriorBlock> meta;
        {
            MargPriorBlock b;
            b.ambient = 7; b.tangent = 6; b.kind = MargBlockKind::POSE;
            b.lin.assign(pose.begin(), pose.end());
            meta.push_back(b);
        }
        for (double *m : {mix_a.data(), mix_b.data()}) {
            MargPriorBlock b;
            b.ambient = 9; b.tangent = 9; b.kind = MargBlockKind::ADDITIVE;
            b.lin.assign(m, m + 9);
            meta.push_back(b);
        }

        ceres::Problem problem;
        problem.AddParameterBlock(pose.data(), 7);
        problem.SetManifold(pose.data(), new PoseManifold());
        problem.AddParameterBlock(mix_a.data(), 9);
        problem.AddParameterBlock(mix_b.data(), 9);
        problem.AddResidualBlock(new MarginalizationPriorFactor(J_out, r_out, meta),
                                 nullptr, pose.data(), mix_a.data(), mix_b.data());

        std::vector<double*> blocks{pose.data(), mix_a.data(), mix_b.data()};
        Eigen::MatrixXd J;
        Eigen::VectorXd r;
        CHECK(evalDense(problem, blocks, {}, J, r), "T1 先验可被 Ceres 求值");

        if (r.size() == 15 && J.rows() == 15) {
            /* 线性化点上 δx = 0 ⇒ 残差应恰为 r_out（这也验证了符号是 + 不是 −） */
            CHECK_NEAR((r - r_out).cwiseAbs().maxCoeff(), 0.0, 1e-12,
                       "T1 线性化点处残差 == r_out（符号约定）");
            /* Ceres 报出的切空间雅可比应恰为 J_out */
            CHECK_NEAR((J - J_out).cwiseAbs().maxCoeff(), 0.0, 1e-12,
                       "T1 切空间雅可比 == J_out");
        } else {
            CHECK(false, "T1 先验维数不对");
        }

        /* 额外做一次有限差分，确认残差函数本身与雅可比自洽：
         * 沿第 k 个切空间方向走 h，残差增量应 ≈ h·J_out[:,k] */
        {
            const int k = 4;
            const double h = 1e-6;
            std::vector<double> pa = pose, mb = mix_a, mc = mix_b;
            /* 切空间方向：从 mix_b 的第 (k-15) 列走（k=4 落在 pose 段） */
            double dp[7];
            memcpy(dp, pose.data(), sizeof(dp));
            /* pose 的切空间方向用 PoseManifold::Plus 实现 */
            PoseManifold pm;
            double delta[6] = {0, 0, 0, 0, 0, 0};
            delta[k] = h;
            double plus[7];
            pm.Plus(pose.data(), delta, plus);
            memcpy(pa.data(), plus, sizeof(plus));

            ceres::Problem p2;
            p2.AddParameterBlock(pa.data(), 7);
            p2.SetManifold(pa.data(), new PoseManifold());
            p2.AddParameterBlock(mb.data(), 9);
            p2.AddParameterBlock(mc.data(), 9);
            p2.AddResidualBlock(new MarginalizationPriorFactor(J_out, r_out, meta),
                                nullptr, pa.data(), mb.data(), mc.data());
            Eigen::MatrixXd J2;
            Eigen::VectorXd r2;
            evalDense(p2, {pa.data(), mb.data(), mc.data()}, {}, J2, r2);
            const double fd = (r2(k) - r_out(k)) / h;
            CHECK_NEAR(fd, J_out(k, k), 1e-4, "T1 有限差分与切空间雅可比一致");
        }
    }

    /* ============================================================================
     * T2：边缘化后的最优解必须与完整问题在保留变量上一致
     * ==========================================================================*/
    static void test_marginalization_equivalence() {
        std::printf("\n---- T2: 边缘化等价性（核心） ----\n");

        /* 量测 */
        const Eigen::Vector3d dp(2.0, 0.3, -0.1), dphi(0.02, -0.01, 0.03), dv(0.5, 0.1, -0.2);
        const double w_coupled = 10.0, w_meas = 1.0;

        const Eigen::Vector3d p0_ref(0.0, 0.0, 0.0), q0_ref(0.0, 0.0, 0.0);
        const Eigen::Vector3d p1_ref(2.1, 0.25, -0.15), q1_ref(0.03, -0.02, 0.04);
        Eigen::VectorXd mix0_ref = Eigen::VectorXd::Zero(9);
        Eigen::VectorXd mix1_ref = Eigen::VectorXd::Zero(9);
        mix0_ref.head(3) << 1.0, 0.0, 0.0;
        mix1_ref.head(3) << 1.4, 0.05, -0.1;

        /* 块初值 */
        std::vector<double> pose0(7, 0.0), mix0(9, 0.0), pose1(7, 0.0), mix1(9, 0.0);
        setPose(pose0.data(), Eigen::Vector3d(0.2, -0.1, 0.05), Eigen::Vector3d(0.01, 0.01, 0.0));
        setPose(pose1.data(), Eigen::Vector3d(2.0, 0.2, -0.1), Eigen::Vector3d(0.02, -0.01, 0.03));
        mix0[0] = 0.9; mix0[1] = 0.02;
        mix1[0] = 1.3; mix1[1] = 0.04;

        auto addFrame = [](ceres::Problem &p, double *pose, double *mix) {
            p.AddParameterBlock(pose, NUM_POSE);
            p.SetManifold(pose, new PoseManifold());
            p.AddParameterBlock(mix, NUM_MIX_ODO);
        };

        /* ---------- 1) 完整问题求解 ---------- */
        ceres::Problem full;
        addFrame(full, pose0.data(), mix0.data());
        addFrame(full, pose1.data(), mix1.data());
        full.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CoupledResidual, 9, 7, 9, 7, 9>(
                new CoupledResidual(dp, dphi, dv, w_coupled)),
            nullptr, pose0.data(), mix0.data(), pose1.data(), mix1.data());
        full.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PoseMeas, 6, 7>(
                new PoseMeas(p0_ref, q0_ref, w_meas)), nullptr, pose0.data());
        full.AddResidualBlock(
            new ceres::AutoDiffCostFunction<MixMeas, 9, 9>(
                new MixMeas(mix0_ref, w_meas)), nullptr, mix0.data());
        full.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PoseMeas, 6, 7>(
                new PoseMeas(p1_ref, q1_ref, w_meas)), nullptr, pose1.data());
        full.AddResidualBlock(
            new ceres::AutoDiffCostFunction<MixMeas, 9, 9>(
                new MixMeas(mix1_ref, w_meas)), nullptr, mix1.data());
        solveProblem(full);

        const std::vector<double> pose1_full = pose1;
        const std::vector<double> mix1_full  = mix1;

        /* ---------- 2) 在完整解处做边缘化（与 GraphOptimizer 同一步骤） ---------- */
        ceres::Problem marg;
        addFrame(marg, pose0.data(), mix0.data());
        addFrame(marg, pose1.data(), mix1.data());
        marg.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CoupledResidual, 9, 7, 9, 7, 9>(
                new CoupledResidual(dp, dphi, dv, w_coupled)),
            nullptr, pose0.data(), mix0.data(), pose1.data(), mix1.data());
        marg.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PoseMeas, 6, 7>(
                new PoseMeas(p0_ref, q0_ref, w_meas)), nullptr, pose0.data());
        marg.AddResidualBlock(
            new ceres::AutoDiffCostFunction<MixMeas, 9, 9>(
                new MixMeas(mix0_ref, w_meas)), nullptr, mix0.data());
        marg.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PoseMeas, 6, 7>(
                new PoseMeas(p1_ref, q1_ref, w_meas)), nullptr, pose1.data());
        marg.AddResidualBlock(
            new ceres::AutoDiffCostFunction<MixMeas, 9, 9>(
                new MixMeas(mix1_ref, w_meas)), nullptr, mix1.data());

        /* 显式列序：frame0 在前 ⇒ 被消去的切空间列是连续前缀 */
        std::vector<double*> ordered{pose0.data(), mix0.data(), pose1.data(), mix1.data()};

        /* 只取触及 frame0 的残差块 —— 这是"不重复计数"的关键 */
        std::vector<ceres::ResidualBlockId> touching;
        for (double *p : {pose0.data(), mix0.data()}) {
            std::vector<ceres::ResidualBlockId> rb;
            marg.GetResidualBlocksForParameterBlock(p, &rb);
            for (auto id : rb) {
                if (std::find(touching.begin(), touching.end(), id) == touching.end())
                    touching.push_back(id);
            }
        }
        CHECK(touching.size() == 3, "T2 只取到 3 个触及 frame0 的残差块");

        Eigen::MatrixXd J;
        Eigen::VectorXd r;
        CHECK(evalDense(marg, ordered, touching, J, r), "T2 边缘化求值成功");

        const int num_marg = 6 + 9;   /* frame0: pose 切空间 6 + mix 9 */
        Eigen::MatrixXd H;
        Eigen::VectorXd b;
        accumulate_normal_equations(J, r, H, b);
        CHECK(H.rows() == 30, "T2 法方程维数 == 30");

        Eigen::MatrixXd J_out;
        Eigen::VectorXd r_out;
        CHECK(schur_complement(H, b, num_marg, J_out, r_out), "T2 Schur 补成功");
        CHECK(J_out.rows() == 15 && J_out.cols() == 15, "T2 先验维数 == 15");

        /* ---------- 3) 组装保留块描述 ---------- */
        std::vector<MargPriorBlock> meta;
        {
            MargPriorBlock bp;
            bp.ambient = 7; bp.tangent = 6; bp.kind = MargBlockKind::POSE;
            bp.lin.assign(pose1.begin(), pose1.end());
            meta.push_back(bp);

            MargPriorBlock bm;
            bm.ambient = 9; bm.tangent = 9; bm.kind = MargBlockKind::ADDITIVE;
            bm.lin.assign(mix1.begin(), mix1.end());
            meta.push_back(bm);
        }

        /* ---------- 4) 约简问题：只留 frame1 的因子 + 先验 ---------- */
        std::vector<double> pose1_red = pose1_full, mix1_red = mix1_full;
        /* 从完整解出发：先验应让它成为驻点 */
        {
            ceres::Problem red;
            addFrame(red, pose1_red.data(), mix1_red.data());
            red.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseMeas, 6, 7>(
                    new PoseMeas(p1_ref, q1_ref, w_meas)), nullptr, pose1_red.data());
            red.AddResidualBlock(
                new ceres::AutoDiffCostFunction<MixMeas, 9, 9>(
                    new MixMeas(mix1_ref, w_meas)), nullptr, mix1_red.data());
            red.AddResidualBlock(new MarginalizationPriorFactor(J_out, r_out, meta),
                                 nullptr, pose1_red.data(), mix1_red.data());
            solveProblem(red);
        }
        {
            double dmax = 0.0;
            for (int k = 0; k < 7; ++k) dmax = std::max(dmax, std::fabs(pose1_red[k] - pose1_full[k]));
            for (int k = 0; k < 9; ++k) dmax = std::max(dmax, std::fabs(mix1_red[k] - mix1_full[k]));
            CHECK_NEAR(dmax, 0.0, 1e-6, "T2 约简问题保持完整解为驻点");
        }

        /* 从扰动起点出发：应收敛回同一个解 */
        {
            std::vector<double> pose1_p = pose1_full, mix1_p = mix1_full;
            pose1_p[0] += 0.8; pose1_p[1] -= 0.5; pose1_p[3] += 0.04;
            mix1_p[0] += 0.6;  mix1_p[4] -= 0.3;

            ceres::Problem red;
            addFrame(red, pose1_p.data(), mix1_p.data());
            red.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseMeas, 6, 7>(
                    new PoseMeas(p1_ref, q1_ref, w_meas)), nullptr, pose1_p.data());
            red.AddResidualBlock(
                new ceres::AutoDiffCostFunction<MixMeas, 9, 9>(
                    new MixMeas(mix1_ref, w_meas)), nullptr, mix1_p.data());
            red.AddResidualBlock(new MarginalizationPriorFactor(J_out, r_out, meta),
                                 nullptr, pose1_p.data(), mix1_p.data());
            solveProblem(red);

            double dmax = 0.0;
            for (int k = 0; k < 7; ++k) dmax = std::max(dmax, std::fabs(pose1_p[k] - pose1_full[k]));
            for (int k = 0; k < 9; ++k) dmax = std::max(dmax, std::fabs(mix1_p[k] - mix1_full[k]));
            CHECK_NEAR(dmax, 0.0, 1e-5, "T2 扰动起点仍收敛到完整解");
        }

        /* ---------- 5) 反向对照：没有先验时解应明显不同（证明先验真在起作用） ---------- */
        {
            std::vector<double> pose1_only = pose1_full, mix1_only = mix1_full;
            ceres::Problem no_prior;
            addFrame(no_prior, pose1_only.data(), mix1_only.data());
            no_prior.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PoseMeas, 6, 7>(
                    new PoseMeas(p1_ref, q1_ref, w_meas)), nullptr, pose1_only.data());
            no_prior.AddResidualBlock(
                new ceres::AutoDiffCostFunction<MixMeas, 9, 9>(
                    new MixMeas(mix1_ref, w_meas)), nullptr, mix1_only.data());
            solveProblem(no_prior);

            double d = 0.0;
            for (int k = 0; k < 7; ++k) d = std::max(d, std::fabs(pose1_only[k] - pose1_full[k]));
            for (int k = 0; k < 9; ++k) d = std::max(d, std::fabs(mix1_only[k] - mix1_full[k]));
            CHECK(d > 1e-3, "T2 对照：去掉先验后解明显不同（先验确实携带信息）");
        }
    }

    /* ============================================================================
     * T3：只触及 frame0 的残差 vs 全部残差 —— 证明"重复计数"会给出不同（错误）的解
     * ==========================================================================*/
    static void test_double_counting_matters() {
        std::printf("\n---- T3: 重复计数会改变解（说明取残差集合不是可有可无） ----\n");

        const Eigen::Vector3d dp(2.0, 0.3, -0.1), dphi(0.02, -0.01, 0.03), dv(0.5, 0.1, -0.2);
        const double w_coupled = 10.0, w_meas = 1.0;
        const Eigen::Vector3d p0_ref(0.0, 0.0, 0.0), q0_ref(0.0, 0.0, 0.0);
        const Eigen::Vector3d p1_ref(2.1, 0.25, -0.15), q1_ref(0.03, -0.02, 0.04);
        Eigen::VectorXd mix0_ref = Eigen::VectorXd::Zero(9), mix1_ref = Eigen::VectorXd::Zero(9);
        mix0_ref.head(3) << 1.0, 0.0, 0.0;
        mix1_ref.head(3) << 1.4, 0.05, -0.1;

        std::vector<double> pose0(7, 0.0), mix0(9, 0.0), pose1(7, 0.0), mix1(9, 0.0);
        setPose(pose0.data(), Eigen::Vector3d(0.2, -0.1, 0.05), Eigen::Vector3d(0.01, 0.01, 0.0));
        setPose(pose1.data(), Eigen::Vector3d(2.0, 0.2, -0.1), Eigen::Vector3d(0.02, -0.01, 0.03));
        mix0[0] = 0.9; mix1[0] = 1.3;

        auto addFrame = [](ceres::Problem &p, double *pose, double *mix) {
            p.AddParameterBlock(pose, NUM_POSE);
            p.SetManifold(pose, new PoseManifold());
            p.AddParameterBlock(mix, NUM_MIX_ODO);
        };

        ceres::Problem pr;
        addFrame(pr, pose0.data(), mix0.data());
        addFrame(pr, pose1.data(), mix1.data());
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<CoupledResidual, 9, 7, 9, 7, 9>(
                                new CoupledResidual(dp, dphi, dv, w_coupled)),
                            nullptr, pose0.data(), mix0.data(), pose1.data(), mix1.data());
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<PoseMeas, 6, 7>(
                                new PoseMeas(p0_ref, q0_ref, w_meas)), nullptr, pose0.data());
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<MixMeas, 9, 9>(
                                new MixMeas(mix0_ref, w_meas)), nullptr, mix0.data());
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<PoseMeas, 6, 7>(
                                new PoseMeas(p1_ref, q1_ref, w_meas)), nullptr, pose1.data());
        pr.AddResidualBlock(new ceres::AutoDiffCostFunction<MixMeas, 9, 9>(
                                new MixMeas(mix1_ref, w_meas)), nullptr, mix1.data());

        std::vector<double*> ordered{pose0.data(), mix0.data(), pose1.data(), mix1.data()};

        /* 只取触及 frame0 的 */
        std::vector<ceres::ResidualBlockId> touching;
        for (double *p : {pose0.data(), mix0.data()}) {
            std::vector<ceres::ResidualBlockId> rb;
            pr.GetResidualBlocksForParameterBlock(p, &rb);
            for (auto id : rb)
                if (std::find(touching.begin(), touching.end(), id) == touching.end())
                    touching.push_back(id);
        }

        Eigen::MatrixXd J1, J2;
        Eigen::VectorXd r1, r2;
        evalDense(pr, ordered, touching, J1, r1);   /* 正确做法 */
        evalDense(pr, ordered, {},       J2, r2);   /* 错误做法：全部残差都进 Schur 补 */

        CHECK(J1.rows() != J2.rows(), "T3 两种取法的残差行数不同");

        Eigen::MatrixXd H1, H2;
        Eigen::VectorXd b1, b2;
        accumulate_normal_equations(J1, r1, H1, b1);
        accumulate_normal_equations(J2, r2, H2, b2);
        Eigen::MatrixXd Jo1, Jo2;
        Eigen::VectorXd ro1, ro2;
        CHECK(schur_complement(H1, b1, 15, Jo1, ro1), "T3 正确取法 Schur 成功");
        CHECK(schur_complement(H2, b2, 15, Jo2, ro2), "T3 错误取法 Schur 成功");

        /* 两种做法产出的先验应当明显不同 —— 说明"残差集合"的选择是实质性的 */
        const double diff = (Jo1 - Jo2).cwiseAbs().maxCoeff() + (ro1 - ro2).cwiseAbs().maxCoeff();
        CHECK(diff > 1e-6, "T3 两种残差集合产出的先验显著不同");
    }

    int run() {
        std::printf("========== 边缘化先验验收测试 ==========\n");
        test_prior_tangent_jacobian();
        test_marginalization_equivalence();
        test_double_counting_matters();
        std::printf("\n=============================================\n");
        std::printf("通过 %d, 失败 %d\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }
}  /* namespace marg_prior */

/* ============================================================================
 * 以下为原 test/test_runner_marg.cpp 的内容，已并入本文件
 * ==========================================================================*/
namespace runner_marg {
    /* ============================================================================
     * test_runner_marg.cpp —— 滑窗边缘化的**端到端**测试
     *
     * 目的：test_marg_prior 验的是算法与因子；这个文件验的是**接线**：
     *   - buildProblem 记录的参数块顺序是否真的和 EvaluateOptions.parameter_blocks 对得上
     *   - resolvePriorPointers 是否能按"帧时间/模糊度键"把先验块解析回当前参数
     *   - optimize() 里先验是否真的被加进了问题（而不是静默丢弃）
     *   - 出窗前调用 marginalizeOldestFrame 之后，解是否仍然合理
     *
     * 仓库里没有任何 .bin 数据，所以这里**合成**一条轨迹：水平姿态、正北匀速 5 m/s、
     * 比力 = −g（于是力学编排里速度恒定），GNSS 位置按 10 Hz 给出。
     * 这样窗口滚动若干次后，若接线有错，通常表现为：解发散、opt_rms 爆掉、
     * 或 n_prior_added 为 0（先验被静默丢掉）。
     * ==========================================================================*/


    static int g_pass = 0;
    static int g_fail = 0;


    /* 确定性伪随机：返回 [-0.5, 0.5) 上的均匀值 */
    static double lcg(uint32_t &st) {
        st = st * 1664525u + 1013904223u;
        return static_cast<double>(st >> 8) * (1.0 / 16777216.0) - 0.5;
    }

    /* 合成一条轨迹：
     *   水平姿态、正北 v = 5 m/s、比力 = −g ⇒ 力学编排下速度恒定、位置线性
     *   GNSS 每 0.1 s 一次（含速度），里程计给出 dS
     * ------------------------------------------------------------------------- */
    static std::vector<DataSensor281_t> makeTrack(double lat0, double lon0, double h0,
                                                  double v_north, double duration,
                                                  double imu_dt) {
        LocalFrame lf;
        lf.init(Eigen::Vector3d(lat0, lon0, h0));
        const Eigen::Vector3d gn = lf.earthModel().gn;      /* NEU 下指向下 */
        const Eigen::Vector3d f_nav = -gn;                  /* 水平匀速 ⇒ 比力 = −g */
        const Eigen::Vector3d v_neu(0.0, v_north, 0.0);

        std::vector<DataSensor281_t> out;
        uint32_t rng_state = 20240918u;
        const int n = static_cast<int>(duration / imu_dt);
        out.reserve(n);
        for (int i = 0; i < n; ++i) {
            const double t = i * imu_dt;
            DataSensor281_t s{};
            s.t = t;
            s.wm[0] = 0.0; s.wm[1] = 0.0; s.wm[2] = 0.0;
            s.vm[0] = f_nav.x() * imu_dt;
            s.vm[1] = f_nav.y() * imu_dt;
            s.vm[2] = f_nav.z() * imu_dt;
            s.dS    = v_north * imu_dt;

            const bool gnss_now = (i % 10 == 0);

            /* 给 GNSS 位置加**确定性**噪声（±0.3 m）：否则真值与量测完全一致，
             * 优化后残差恒为 0，"小窗口/大窗口精度比"这个检验就没有意义了。
             * 用自带的 LCG 而不是 <random>，保证跨平台可复现。 */
            Eigen::Vector3d neu_noisy = v_neu * t;
            if (gnss_now) {
                neu_noisy += Eigen::Vector3d(0.6 * lcg(rng_state),
                                             0.6 * lcg(rng_state),
                                             0.6 * lcg(rng_state));
            }
            const Eigen::Vector3d blh = lf.neu2blh(neu_noisy);

            s.flagGNSS = gnss_now ? 1.0 : 0.0;
            s.posgps[0] = blh.x();
            s.posgps[1] = blh.y();
            s.posgps[2] = blh.z();
            s.vngps[0] = 0.0; s.vngps[1] = v_north; s.vngps[2] = 0.0;
            s.satnum   = 12.0;
            s.baseline = 0.0;
            s.yaw      = 0.0;
            s.yawrms   = 179.99;        /* 无效标志：不使用双天线航向 */
            s.posstd[0] = 0.3; s.posstd[1] = 0.3; s.posstd[2] = 0.6;
            out.push_back(s);
        }
        return out;
    }

    static RunnerOptions makeOptions(int max_kf, double kf_dt) {
        RunnerOptions o;
        o.kf_dt              = kf_dt;
        o.max_keyframes      = max_kf;
        o.use_gnss_vel       = true;
        o.use_gnss_yaw       = false;
        o.use_static         = false;
        o.init_yaw_from_gnss = false;
        o.fix_first_pose     = true;
        o.use_odometer           = false;
        return o;
    }

    int run() {
        std::printf("========== 滑窗边缘化端到端测试 ==========\n");

        const double lat0 = 30.0 * DEG, lon0 = 114.0 * DEG, h0 = 20.0;
        const double v_north = 5.0;
        const double duration = 20.0;      /* s */
        const double imu_dt = 0.02;        /* 50 Hz（够用即可，端到端测试要跑得快） */

        const auto track = makeTrack(lat0, lon0, h0, v_north, duration, imu_dt);
        std::printf("合成轨迹：%zu 帧，%.1f s，正北 %.1f m/s\n",
                    track.size(), duration, v_north);

        /* ---------- A) 小窗口：一定会发生出窗 ⇒ 边缘化 ---------- */
        const int kf_small = 5;
        RunnerStats a;
        {
            RunnerOptions o = makeOptions(kf_small, 1.0);
            a = runRealData(track, o, "");
        }
        std::printf("小窗口: n_frames=%d n_kf_total=%d n_kf=%d n_prior_added=%d opt_rms=%.4f\n",
                    a.n_frames, a.n_kf_total, a.n_kf, a.n_prior_added, a.opt_rms);

        CHECK(a.n_frames == static_cast<int>(track.size()), "A 输入帧数一致");
        CHECK(a.n_kf_total > kf_small, "A 关键帧总数超过窗口上限（确实发生了出窗）");
        CHECK(a.n_kf <= kf_small, "A 窗口内关键帧未超上限");
        CHECK(a.n_prior_added > 0, "A 边缘化先验被成功加入问题（接线生效）");
        CHECK(std::isfinite(a.opt_rms), "A 优化后 RMS 有限");
        CHECK(std::isfinite(a.opt_max), "A 优化后最大误差有限");
        CHECK(a.opt_rms < 5.0, "A 优化后 RMS 在合理范围（先验没有把解带歪）");
        CHECK(a.improve_ratio > 0.0, "A improve_ratio 为正");

        /* ---------- B) 大窗口：基本不出窗（对照） ---------- */
        RunnerStats b;
        {
            RunnerOptions o = makeOptions(100, 1.0);
            b = runRealData(track, o, "");
        }
        std::printf("大窗口: n_kf_total=%d n_kf=%d n_prior_added=%d opt_rms=%.4f\n",
                    b.n_kf_total, b.n_kf, b.n_prior_added, b.opt_rms);

        CHECK(b.n_kf == b.n_kf_total, "B 大窗口未发生出窗（对照组）");
        CHECK(b.opt_rms < 5.0, "B 大窗口 RMS 合理");

        /* ---------- C) 两者精度应当可比：边缘化不应显著劣化 ---------- */
        if (std::isfinite(a.opt_rms) && std::isfinite(b.opt_rms) && b.opt_rms > 1e-9) {
            const double ratio = a.opt_rms / b.opt_rms;
            std::printf("小窗口/大窗口 RMS 比 = %.3f\n", ratio);
            CHECK(ratio < 5.0, "C 边缘化后的精度与全窗口同量级（未显著劣化）");
        } else {
            CHECK(false, "C 两组 RMS 都应有限且非零");
        }

        /* ---------- D) fix_first_pose=false 的分支也要能跑通 ---------- */
        {
            RunnerOptions o = makeOptions(kf_small, 1.0);
            o.fix_first_pose = false;
            RunnerStats d = runRealData(track, o, "");
            std::printf("不固定首帧: n_prior_added=%d opt_rms=%.4f\n", d.n_prior_added, d.opt_rms);
            CHECK(std::isfinite(d.opt_rms), "D fix_first_pose=false 可跑通");
            CHECK(d.opt_rms < 20.0, "D fix_first_pose=false 解仍有界");
        }

        std::printf("\n=============================================\n");
        std::printf("通过 %d, 失败 %d\n", g_pass, g_fail);
        return g_fail == 0 ? 0 : 1;
    }
}  /* namespace runner_marg */

/* ============================================================================
 * 汇总入口
 * ==========================================================================*/
int main() {
    std::printf("############ test_marg（Schur 补 + 边缘化先验 + runner 边缘化）############\n");
    int bad = 0;
    bad += marg_base::run();
    bad += marg_prior::run();
    bad += runner_marg::run();

    const int P = marg_base::g_pass + marg_prior::g_pass + runner_marg::g_pass;
    const int F = marg_base::g_fail + marg_prior::g_fail + runner_marg::g_fail;
    std::printf("\n================ test_marg 汇总 ================\n");
    std::printf("总断言: %d, 通过: %d, 失败: %d\n", P + F, P, F);
    return bad == 0 ? 0 : 1;
}
