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
 * vect3
 * ==========================================================================*/

static void test_vect3_constructors() {
    vect3 a;
    EXPECT_NEAR(a.i, 0.0, 1e-15);
    EXPECT_NEAR(a.j, 0.0, 1e-15);
    EXPECT_NEAR(a.k, 0.0, 1e-15);

    vect3 b(2.5);
    EXPECT_NEAR(b.i, 2.5, 1e-15);
    EXPECT_NEAR(b.j, 2.5, 1e-15);
    EXPECT_NEAR(b.k, 2.5, 1e-15);

    vect3 c(1.0, 2.0, 3.0);
    EXPECT_NEAR(c.i, 1.0, 1e-15);
    EXPECT_NEAR(c.j, 2.0, 1e-15);
    EXPECT_NEAR(c.k, 3.0, 1e-15);

    double arr[3] = {4.0, 5.0, 6.0};
    vect3 d(arr);
    EXPECT_NEAR(d.i, 4.0, 1e-15);
    EXPECT_NEAR(d.j, 5.0, 1e-15);
    EXPECT_NEAR(d.k, 6.0, 1e-15);

    float farr[3] = {7.0f, 8.0f, 9.0f};
    vect3 e(farr);
    EXPECT_NEAR(e.i, 7.0, 1e-6);
    EXPECT_NEAR(e.j, 8.0, 1e-6);
    EXPECT_NEAR(e.k, 9.0, 1e-6);

    Eigen::Vector3d ev(1.5, -2.5, 3.5);
    vect3 f(ev);
    EXPECT_NEAR(f.i, 1.5, 1e-15);
    EXPECT_NEAR(f.j, -2.5, 1e-15);
    EXPECT_NEAR(f.k, 3.5, 1e-15);
}

static void test_vect3_eigen_roundtrip() {
    vect3 a(1.0, 2.0, 3.0);
    Eigen::Vector3d e = a.toEigen();
    vect3 b(e);
    EXPECT_NEAR(b.i, a.i, 1e-15);
    EXPECT_NEAR(b.j, a.j, 1e-15);
    EXPECT_NEAR(b.k, a.k, 1e-15);

    /* e() 与 toEigen() 一致性 */
    const vect3 &ca = a;
    Eigen::Vector3d from_map = ca.e();
    EXPECT_NEAR(from_map(0), 1.0, 1e-15);
    EXPECT_NEAR(from_map(1), 2.0, 1e-15);
    EXPECT_NEAR(from_map(2), 3.0, 1e-15);

    /* 可写 e()：验证内存共享 */
    Eigen::Map<Eigen::Vector3d> m = a.e();
    m(0) = 42.0;
    EXPECT_NEAR(a.i, 42.0, 1e-15);
}

static void test_vect3_arithmetic() {
    vect3 a(1.0, 2.0, 3.0), b(4.0, 5.0, 6.0);

    vect3 s = a + b;
    EXPECT_NEAR(s.i, 5.0, 1e-15);
    EXPECT_NEAR(s.j, 7.0, 1e-15);
    EXPECT_NEAR(s.k, 9.0, 1e-15);

    vect3 d = a - b;
    EXPECT_NEAR(d.i, -3.0, 1e-15);
    EXPECT_NEAR(d.j, -3.0, 1e-15);
    EXPECT_NEAR(d.k, -3.0, 1e-15);

    vect3 p = a * 2.0;
    EXPECT_NEAR(p.i, 2.0, 1e-15);
    EXPECT_NEAR(p.j, 4.0, 1e-15);
    EXPECT_NEAR(p.k, 6.0, 1e-15);

    vect3 p2 = 2.0 * a;
    EXPECT_NEAR(p2.i, 2.0, 1e-15);
    EXPECT_NEAR(p2.j, 4.0, 1e-15);
    EXPECT_NEAR(p2.k, 6.0, 1e-15);

    vect3 q = a / 2.0;
    EXPECT_NEAR(q.i, 0.5, 1e-15);
    EXPECT_NEAR(q.j, 1.0, 1e-15);
    EXPECT_NEAR(q.k, 1.5, 1e-15);

    vect3 qq = a / b;
    EXPECT_NEAR(qq.i, 0.25, 1e-15);
    EXPECT_NEAR(qq.j, 0.4, 1e-15);
    EXPECT_NEAR(qq.k, 0.5, 1e-15);

    vect3 neg = -a;
    EXPECT_NEAR(neg.i, -1.0, 1e-15);
    EXPECT_NEAR(neg.j, -2.0, 1e-15);
    EXPECT_NEAR(neg.k, -3.0, 1e-15);
}

static void test_vect3_cross() {
    /* 右手定则 */
    vect3 x(1.0, 0.0, 0.0), y(0.0, 1.0, 0.0), z(0.0, 0.0, 1.0);
    vect3 xy = x * y;
    EXPECT_NEAR(xy.i, 0.0, 1e-15);
    EXPECT_NEAR(xy.j, 0.0, 1e-15);
    EXPECT_NEAR(xy.k, 1.0, 1e-15);

    vect3 yz = y * z;
    EXPECT_NEAR(yz.i, 1.0, 1e-15);
    EXPECT_NEAR(yz.j, 0.0, 1e-15);
    EXPECT_NEAR(yz.k, 0.0, 1e-15);

    /* 反对称性：a×b = -b×a */
    vect3 a(1.0, 2.0, 3.0), b(4.0, 5.0, 6.0);
    vect3 ab = a * b, ba = b * a;
    EXPECT_NEAR(ab.i, -ba.i, 1e-15);
    EXPECT_NEAR(ab.j, -ba.j, 1e-15);
    EXPECT_NEAR(ab.k, -ba.k, 1e-15);

    /* 自叉乘为零 */
    vect3 aa = a * a;
    EXPECT_NEAR(norm(aa), 0.0, 1e-15);
}

static void test_vect3_compound_assign() {
    vect3 a(1.0, 2.0, 3.0);
    a += vect3(1.0, 1.0, 1.0);
    EXPECT_NEAR(a.i, 2.0, 1e-15);
    EXPECT_NEAR(a.k, 4.0, 1e-15);

    a -= vect3(0.5, 0.5, 0.5);
    EXPECT_NEAR(a.i, 1.5, 1e-15);
    EXPECT_NEAR(a.k, 3.5, 1e-15);

    a *= 2.0;
    EXPECT_NEAR(a.i, 3.0, 1e-15);
    EXPECT_NEAR(a.k, 7.0, 1e-15);

    a /= 2.0;
    EXPECT_NEAR(a.i, 1.5, 1e-15);
    EXPECT_NEAR(a.k, 3.5, 1e-15);

    a /= vect3(1.5, 1.5, 1.5);
    EXPECT_NEAR(a.i, 1.0, 1e-15);
    EXPECT_NEAR(a.j, 5.0 / 3.0, 1e-15);
    EXPECT_NEAR(a.k, 7.0 / 3.0, 1e-15);
}

static void test_vect3_queries() {
    vect3 a(1e-20, -1e-20, 0.0);
    EXPECT_TRUE(IsZeros(a));
    EXPECT_TRUE(IsZero(a.i));
    EXPECT_TRUE(IsZerosXY(a));

    vect3 b(0.5, 0.0, 0.0);
    EXPECT_TRUE(!IsZeros(b));
    EXPECT_TRUE(!IsZerosXY(b));  /* j 分量虽为 0，但 i 不满足 */

    vect3 c(NAN, 0.0, 0.0);
    EXPECT_TRUE(IsNaN(c));

    vect3 d(0.0, 0.0, 0.0);
    EXPECT_TRUE(!IsNaN(d));
    EXPECT_TRUE(IsZeros(d));
}

static void test_vect3_norms() {
    vect3 a(3.0, 4.0, 0.0);
    EXPECT_NEAR(norm(a), 5.0, 1e-15);
    EXPECT_NEAR(normXY(a), 5.0, 1e-15);
    EXPECT_NEAR(normInf(a), 4.0, 1e-15);
    EXPECT_NEAR(normXYInf(a), 4.0, 1e-15);

    vect3 b(0.0);
    EXPECT_NEAR(norm(b), 0.0, 1e-15);
    EXPECT_NEAR(normXY(b), 0.0, 1e-15);
}

static void test_vect3_dot_and_products() {
    vect3 a(1.0, 2.0, 3.0), b(4.0, 5.0, 6.0);
    EXPECT_NEAR(dot(a, b), 32.0, 1e-15);
    EXPECT_NEAR(dot(a, a), 14.0, 1e-15);

    vect3 e = dotmul(a, b);
    EXPECT_NEAR(e.i, 4.0, 1e-15);
    EXPECT_NEAR(e.j, 10.0, 1e-15);
    EXPECT_NEAR(e.k, 18.0, 1e-15);

    /* 正交向量夹角正弦 */
    vect3 x(1, 0, 0), y(0, 1, 0);
    EXPECT_NEAR(sinAng(x, y), 1.0, 1e-15);
    EXPECT_NEAR(sinAng(x, x), 0.0, 1e-15);

    /* vxv 外积 */
    mat3 m = vxv(x, y);
    EXPECT_NEAR(m.e00, 0.0, 1e-15);
    EXPECT_NEAR(m.e01, 1.0, 1e-15);
    EXPECT_NEAR(m.e10, 0.0, 1e-15);
    EXPECT_NEAR(m.e11, 0.0, 1e-15);
}

static void test_vect3_elementwise() {
    vect3 a(-1.0, 4.0, -9.0);
    vect3 s = sqrt(abs(a));
    EXPECT_NEAR(s.i, 1.0, 1e-15);
    EXPECT_NEAR(s.j, 2.0, 1e-15);
    EXPECT_NEAR(s.k, 3.0, 1e-15);

    vect3 b(2.0, 3.0, 4.0);
    vect3 p2 = pow(b, 2);
    EXPECT_NEAR(p2.i, 4.0, 1e-15);
    EXPECT_NEAR(p2.j, 9.0, 1e-15);
    EXPECT_NEAR(p2.k, 16.0, 1e-15);

    vect3 p3 = pow(b, 3);
    EXPECT_NEAR(p3.i, 8.0, 1e-15);
    EXPECT_NEAR(p3.j, 27.0, 1e-15);
    EXPECT_NEAR(p3.k, 64.0, 1e-15);

    vect3 m1(1.0, 5.0, 3.0), m2(-2.0, 4.0, -6.0);
    vect3 mx = maxabs(m1, m2);
    EXPECT_NEAR(mx.i, 2.0, 1e-15);
    EXPECT_NEAR(mx.j, 5.0, 1e-15);
    EXPECT_NEAR(mx.k, 6.0, 1e-15);

    vect3 srt = sort(vect3(1.0, 5.0, 3.0));
    EXPECT_NEAR(srt.i, 5.0, 1e-15);
    EXPECT_NEAR(srt.j, 3.0, 1e-15);
    EXPECT_NEAR(srt.k, 1.0, 1e-15);
}

/* ============================================================================
 * mat3
 * ==========================================================================*/

static void test_mat3_constructors() {
    mat3 z;
    EXPECT_NEAR(z.e00, 0.0, 1e-15);
    EXPECT_NEAR(z.e22, 0.0, 1e-15);

    mat3 d(2.0, 3.0, 4.0);
    EXPECT_NEAR(d.e00, 2.0, 1e-15);
    EXPECT_NEAR(d.e11, 3.0, 1e-15);
    EXPECT_NEAR(d.e22, 4.0, 1e-15);
    EXPECT_NEAR(d.e01, 0.0, 1e-15);

    mat3 m(1, 2, 3, 4, 5, 6, 7, 8, 9);
    EXPECT_NEAR(m.e00, 1.0, 1e-15);
    EXPECT_NEAR(m.e01, 2.0, 1e-15);
    EXPECT_NEAR(m.e22, 9.0, 1e-15);

    double arr[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    mat3 b(arr);
    EXPECT_NEAR(b.e00, 1.0, 1e-15);
    EXPECT_NEAR(b.e21, 8.0, 1e-15);

    Eigen::Matrix3d em;
    em << 10, 20, 30, 40, 50, 60, 70, 80, 90;
    mat3 c(em);
    EXPECT_NEAR(c.e00, 10.0, 1e-15);
    EXPECT_NEAR(c.e11, 50.0, 1e-15);
    EXPECT_NEAR(c.e22, 90.0, 1e-15);
}

static void test_mat3_rows_columns() {
    /* 用三个向量按行构造 */
    vect3 r0(1, 2, 3), r1(4, 5, 6), r2(7, 8, 9);
    mat3 m(r0, r1, r2, true);
    EXPECT_NEAR(m.e00, 1.0, 1e-15);
    EXPECT_NEAR(m.e01, 2.0, 1e-15);
    EXPECT_NEAR(m.e10, 4.0, 1e-15);
    EXPECT_NEAR(m.e22, 9.0, 1e-15);

    /* 按列构造 */
    mat3 mc(r0, r1, r2, false);
    EXPECT_NEAR(mc.e00, 1.0, 1e-15);
    EXPECT_NEAR(mc.e01, 4.0, 1e-15);
    EXPECT_NEAR(mc.e02, 7.0, 1e-15);
    EXPECT_NEAR(mc.e10, 2.0, 1e-15);

    /* SetRow / GetRow */
    mat3 m2;
    m2.SetRow(0, r0);
    m2.SetRow(1, r1);
    m2.SetRow(2, r2);
    vect3 g0 = m2.GetRow(0);
    vect3 g2 = m2.GetRow(2);
    EXPECT_NEAR(g0.i, 1.0, 1e-15);
    EXPECT_NEAR(g0.k, 3.0, 1e-15);
    EXPECT_NEAR(g2.i, 7.0, 1e-15);
    EXPECT_NEAR(g2.k, 9.0, 1e-15);

    /* SetClm / GetClm */
    mat3 m3;
    m3.SetClm(0, r0);
    m3.SetClm(1, r1);
    m3.SetClm(2, r2);
    vect3 c1 = m3.GetClm(1);
    EXPECT_NEAR(c1.i, 4.0, 1e-15);
    EXPECT_NEAR(c1.j, 5.0, 1e-15);
    EXPECT_NEAR(c1.k, 6.0, 1e-15);
}

static void test_mat3_arithmetic() {
    mat3 A(1, 2, 3, 4, 5, 6, 7, 8, 9);
    mat3 B(9, 8, 7, 6, 5, 4, 3, 2, 1);

    mat3 S = A + B;
    EXPECT_NEAR(S.e00, 10.0, 1e-15);
    EXPECT_NEAR(S.e22, 10.0, 1e-15);

    mat3 D = A - B;
    EXPECT_NEAR(D.e00, -8.0, 1e-15);
    EXPECT_NEAR(D.e22, 8.0, 1e-15);

    mat3 M = A * 2.0;
    EXPECT_NEAR(M.e00, 2.0, 1e-15);
    EXPECT_NEAR(M.e22, 18.0, 1e-15);

    mat3 M2 = 2.0 * A;
    EXPECT_NEAR(M2.e00, 2.0, 1e-15);
    EXPECT_NEAR(M2.e22, 18.0, 1e-15);

    /* 转置 */
    mat3 T = ~A;
    EXPECT_NEAR(T.e01, A.e10, 1e-15);
    EXPECT_NEAR(T.e10, A.e01, 1e-15);
    EXPECT_NEAR(T.e20, A.e02, 1e-15);

    /* 加法到对角线 */
    mat3 D2 = A + vect3(10, 20, 30);
    EXPECT_NEAR(D2.e00, 11.0, 1e-15);
    EXPECT_NEAR(D2.e11, 25.0, 1e-15);
    EXPECT_NEAR(D2.e22, 39.0, 1e-15);
    EXPECT_NEAR(D2.e01, A.e01, 1e-15);
}

static void test_mat3_multiplication() {
    /* 单位阵 */
    mat3 A(1, 2, 3, 4, 5, 6, 7, 8, 9);
    mat3 P = A * I33;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(P(i, j), A(i, j), 1e-15);
        }
    }

    /* 手算 2x2 嵌入 */
    mat3 M(1, 2, 0, 3, 4, 0, 0, 0, 1);
    mat3 N(5, 6, 0, 7, 8, 0, 0, 0, 1);
    mat3 R = M * N;
    EXPECT_NEAR(R.e00, 19.0, 1e-14);
    EXPECT_NEAR(R.e01, 22.0, 1e-14);
    EXPECT_NEAR(R.e10, 43.0, 1e-14);
    EXPECT_NEAR(R.e11, 50.0, 1e-14);

    /* mat3 * vect3 */
    vect3 v(1.0, 1.0, 1.0);
    vect3 Av = A * v;
    EXPECT_NEAR(Av.i, 6.0, 1e-14);
    EXPECT_NEAR(Av.j, 15.0, 1e-14);
    EXPECT_NEAR(Av.k, 24.0, 1e-14);
}

static void test_mat3_trace_det_inv() {
    mat3 A(1, 2, 3, 4, 5, 6, 7, 8, 10);
    EXPECT_NEAR(trace(A), 16.0, 1e-14);

    /* det(A) = 1*(5*10-6*8) - 2*(4*10-6*7) + 3*(4*8-5*7) = 2 + 4 - 9 = -3 */
    EXPECT_NEAR(det(A), -3.0, 1e-12);

    /* A * inv(A) ≈ I */
    mat3 Ai = inv(A);
    mat3 P = A * Ai;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(P(i, j), (i == j) ? 1.0 : 0.0, 1e-12);
        }
    }

    /* 正交阵的逆等于转置 */
    vect3 rv(0.1, 0.2, 0.3);
    mat3 R = rv2m(rv);
    mat3 Ri = inv(R);
    mat3 Rt = ~R;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(Ri(i, j), Rt(i, j), 1e-12);
        }
    }

    /* adj(A) = det(A) * inv(A) */
    mat3 Adj = adj(A);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(Adj(i, j), det(A) * Ai(i, j), 1e-12);
        }
    }
}

static void test_mat3_askew() {
    /* askew(v) * w = v × w */
    vect3 v(1.0, 2.0, 3.0), w(4.0, 5.0, 6.0);
    mat3 S = askew(v);
    vect3 vw = v * w;
    vect3 Svw = S * w;
    EXPECT_NEAR(Svw.i, vw.i, 1e-14);
    EXPECT_NEAR(Svw.j, vw.j, 1e-14);
    EXPECT_NEAR(Svw.k, vw.k, 1e-14);

    /* 反对称性 */
    EXPECT_NEAR(S.e01, -S.e10, 1e-15);
    EXPECT_NEAR(S.e02, -S.e20, 1e-15);
    EXPECT_NEAR(S.e12, -S.e21, 1e-15);
    EXPECT_NEAR(S.e00, 0.0, 1e-15);
}

static void test_mat3_rot() {
    /* Rx(π/2): y -> z, z -> -y */
    mat3 Rx = Rot(PI / 2, 'x');
    vect3 y(0, 1, 0), z(0, 0, 1);
    vect3 Rxy = Rx * y, Rxz = Rx * z;
    EXPECT_NEAR(Rxy.k, 1.0, 1e-15);
    EXPECT_NEAR(Rxy.j, 0.0, 1e-15);
    EXPECT_NEAR(Rxz.j, -1.0, 1e-15);

    /* 正交性：R^T R = I */
    for (char axis : {'x', 'y', 'z'}) {
        mat3 R = Rot(0.7, axis);
        mat3 RtR = (~R) * R;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                EXPECT_NEAR(RtR(i, j), (i == j) ? 1.0 : 0.0, 1e-14);
            }
        }
        EXPECT_NEAR(det(R), 1.0, 1e-14);
    }
}

static void test_mat3_diag() {
    mat3 A(1, 2, 3, 4, 5, 6, 7, 8, 9);
    vect3 d = diag(A);
    EXPECT_NEAR(d.i, 1.0, 1e-15);
    EXPECT_NEAR(d.j, 5.0, 1e-15);
    EXPECT_NEAR(d.k, 9.0, 1e-15);

    mat3 D = diag(vect3(2, 4, 6));
    EXPECT_NEAR(D.e00, 2.0, 1e-15);
    EXPECT_NEAR(D.e11, 4.0, 1e-15);
    EXPECT_NEAR(D.e22, 6.0, 1e-15);
    EXPECT_NEAR(D.e01, 0.0, 1e-15);
}

static void test_mat3_dotmul_and_norm() {
    mat3 A(1, 2, 3, 4, 5, 6, 7, 8, 9);
    mat3 B(2, 2, 2, 2, 2, 2, 2, 2, 2);
    mat3 E = dotmul(A, B);
    EXPECT_NEAR(E.e00, 2.0, 1e-15);
    EXPECT_NEAR(E.e11, 10.0, 1e-15);
    EXPECT_NEAR(E.e22, 18.0, 1e-15);

    /* norm(A) = sqrt(tr(A A^T)) */
    double expect = std::sqrt(1.0 + 4.0 + 9.0 + 16.0 + 25.0 + 36.0 + 49.0 + 64.0 + 81.0);
    EXPECT_NEAR(norm(A), expect, 1e-14);

    /* trMMT(A, I) = tr(A) */
    EXPECT_NEAR(trMMT(A), norm(A) * norm(A), 1e-12);  
    EXPECT_NEAR(trMMT(A), 285.0, 1e-12); 
}

static void test_mat3_pow() {
    mat3 A(1, 1, 0, 0, 1, 1, 0, 0, 1);
    mat3 A2 = pow(A, 2);
    mat3 A2_expect = A * A;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(A2(i, j), A2_expect(i, j), 1e-14);
        }
    }

    mat3 I = pow(A, 1);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(I(i, j), A(i, j), 1e-14);
        }
    }
}

/* ============================================================================
 * quat
 * ==========================================================================*/

static void test_quat_constructors() {
    quat q0;
    EXPECT_NEAR(q0.q0, 0.0, 1e-15);

    quat q1(1.0);
    EXPECT_NEAR(q1.q0, 1.0, 1e-15);
    EXPECT_NEAR(q1.q1, 0.0, 1e-15);

    quat q2(1.0, 2.0, 3.0, 4.0);
    EXPECT_NEAR(q2.q0, 1.0, 1e-15);
    EXPECT_NEAR(q2.q3, 4.0, 1e-15);

    quat q3(0.5, vect3(1.0, 2.0, 3.0));
    EXPECT_NEAR(q3.q0, 0.5, 1e-15);
    EXPECT_NEAR(q3.q1, 1.0, 1e-15);
    EXPECT_NEAR(q3.q2, 2.0, 1e-15);
    EXPECT_NEAR(q3.q3, 3.0, 1e-15);

    double arr[4] = {5.0, 6.0, 7.0, 8.0};
    quat q4(arr);
    EXPECT_NEAR(q4.q0, 5.0, 1e-15);
    EXPECT_NEAR(q4.q3, 8.0, 1e-15);

    Eigen::Quaterniond eq(1.0, 0.1, 0.2, 0.3);
    quat q5(eq);
    EXPECT_NEAR(q5.q0, 1.0, 1e-15);
    EXPECT_NEAR(q5.q1, 0.1, 1e-15);
    EXPECT_NEAR(q5.q2, 0.2, 1e-15);
    EXPECT_NEAR(q5.q3, 0.3, 1e-15);
}

static void test_quat_eigen_roundtrip() {
    quat q(0.5, 0.5, 0.5, 0.5);
    Eigen::Quaterniond e = q.toEigen();
    quat q2(e);
    EXPECT_NEAR(q2.q0, q.q0, 1e-15);
    EXPECT_NEAR(q2.q1, q.q1, 1e-15);
    EXPECT_NEAR(q2.q2, q.q2, 1e-15);
    EXPECT_NEAR(q2.q3, q.q3, 1e-15);
}

static void test_quat_multiplication() {
    /* 单位元 */
    quat qI(1.0, 0.0, 0.0, 0.0);
    quat q(0.5, 0.5, 0.5, 0.5);
    normlize(&q);
    quat p = q * qI;
    EXPECT_NEAR(p.q0, q.q0, 1e-15);
    EXPECT_NEAR(p.q1, q.q1, 1e-15);

    quat p2 = qI * q;
    EXPECT_NEAR(p2.q0, q.q0, 1e-15);
    EXPECT_NEAR(p2.q1, q.q1, 1e-15);

    /* 旋转的复合 = 四元数相乘 */
    vect3 rv1(0.1, 0.0, 0.0), rv2(0.0, 0.2, 0.0);
    quat q1 = rv2q(rv1), q2 = rv2q(rv2);
    quat q12 = q1 * q2;

    /* 分别作用在向量上 */
    vect3 v(1.0, 0.0, 0.0);
    vect3 v1 = q1 * (q2 * v);
    vect3 v2 = q12 * v;
    EXPECT_NEAR(v1.i, v2.i, 1e-14);
    EXPECT_NEAR(v1.j, v2.j, 1e-14);
    EXPECT_NEAR(v1.k, v2.k, 1e-14);
}

static void test_quat_inverse() {
    quat q(0.5, 0.5, 0.5, 0.5);
    normlize(&q);
    quat qi = ~q;
    quat p = q * qi;
    EXPECT_NEAR(p.q0, 1.0, 1e-14);
    EXPECT_NEAR(p.q1, 0.0, 1e-14);
    EXPECT_NEAR(p.q2, 0.0, 1e-14);
    EXPECT_NEAR(p.q3, 0.0, 1e-14);
}

static void test_quat_rotate_vector() {
    /* Rx(π/2): y -> z */
    quat q = rv2q(vect3(PI / 2, 0, 0));
    vect3 v(0, 1, 0);
    vect3 r = q * v;
    EXPECT_NEAR(r.i, 0.0, 1e-14);
    EXPECT_NEAR(r.j, 0.0, 1e-14);
    EXPECT_NEAR(r.k, 1.0, 1e-14);

    /* 与 mat3 路径一致 */
    mat3 R = q2mat(q);
    vect3 r2 = R * v;
    EXPECT_NEAR(r.i, r2.i, 1e-14);
    EXPECT_NEAR(r.j, r2.j, 1e-14);
    EXPECT_NEAR(r.k, r2.k, 1e-14);
}

static void test_quat_compound_assign() {
    quat q(1.0);
    quat qa = rv2q(vect3(0.1, 0.0, 0.0));
    quat qb = rv2q(vect3(0.0, 0.2, 0.0));
    q *= qa;
    q *= qb;
    quat ref = qa * qb;
    EXPECT_NEAR(q.q0, ref.q0, 1e-14);
    EXPECT_NEAR(q.q1, ref.q1, 1e-14);
    EXPECT_NEAR(q.q2, ref.q2, 1e-14);
    EXPECT_NEAR(q.q3, ref.q3, 1e-14);
}

static void test_quat_setYaw() {
    quat q = a2qua(vect3(0.1, 0.2, 0.3));
    double yaw_new = 0.7;
    q.SetYaw(yaw_new);
    vect3 att = q2att(q);
    EXPECT_NEAR(att.k, yaw_new, 1e-12);
    EXPECT_NEAR(att.i, 0.1, 1e-12);
    EXPECT_NEAR(att.j, 0.2, 1e-12);
}

/* ============================================================================
 * 姿态/旋转函数
 * ==========================================================================*/

static void test_rv2q_q2rv_roundtrip() {
    /* 小角 */
    vect3 rv(0.01, -0.02, 0.03);
    quat q = rv2q(rv);
    vect3 rv2 = q2rv(q);
    EXPECT_NEAR(rv2.i, rv.i, 1e-13);
    EXPECT_NEAR(rv2.j, rv.j, 1e-13);
    EXPECT_NEAR(rv2.k, rv.k, 1e-13);

    /* 中角 */
    vect3 rv_m(0.5, -0.3, 0.7);
    quat qm = rv2q(rv_m);
    vect3 rv2_m = q2rv(qm);
    EXPECT_NEAR(rv2_m.i, rv_m.i, 1e-12);
    EXPECT_NEAR(rv2_m.j, rv_m.j, 1e-12);
    EXPECT_NEAR(rv2_m.k, rv_m.k, 1e-12);

    /* 零角 */
    vect3 zero(0, 0, 0);
    quat qz = rv2q(zero);
    EXPECT_NEAR(qz.q0, 1.0, 1e-15);
    EXPECT_NEAR(norm(q2rv(qz)), 0.0, 1e-14);
}

static void test_rv2q_small_angle_taylor() {
    /* 小角度应满足 |rv| / 2 的近似 */
    double eps = 1e-6;
    vect3 rv(eps, 0, 0);
    quat q = rv2q(rv);
    /* q ≈ [1, rv/2] */
    EXPECT_NEAR(q.q0, 1.0, 1e-12);
    EXPECT_NEAR(q.q1, eps / 2.0, 1e-14);
    EXPECT_NEAR(q.q2, 0.0, 1e-15);
    EXPECT_NEAR(q.q3, 0.0, 1e-15);
}

static void test_q2mat_m2qua_roundtrip() {
    vect3 rv(0.2, -0.4, 0.6);
    quat q1 = rv2q(rv);
    mat3 R  = q2mat(q1);
    quat q2 = m2qua(R);
    /* q2 = ±q1；允许符号翻转 */
    double dot = q1.q0 * q2.q0 + q1.q1 * q2.q1 + q1.q2 * q2.q2 + q1.q3 * q2.q3;
    EXPECT_NEAR(std::fabs(dot), 1.0, 1e-12);
}

static void test_a2qua_q2att_roundtrip() {
    vect3 att(0.1, -0.2, 0.3);
    quat q = a2qua(att);
    vect3 att2 = q2att(q);
    EXPECT_NEAR(att2.i, att.i, 1e-12);
    EXPECT_NEAR(att2.j, att.j, 1e-12);
    EXPECT_NEAR(att2.k, att.k, 1e-12);
}

static void test_a2mat_m2att_roundtrip() {
    vect3 att(0.05, -0.15, 1.2);
    mat3 R = a2mat(att);
    vect3 att2 = m2att(R);
    EXPECT_NEAR(att2.i, att.i, 1e-12);
    EXPECT_NEAR(att2.j, att.j, 1e-12);
    EXPECT_NEAR(att2.k, att.k, 1e-12);
}

static void test_a2qua_matches_a2mat() {
    vect3 att(0.1, 0.2, 0.3);
    quat q = a2qua(att);
    mat3 Rq = q2mat(q);
    mat3 Ra = a2mat(att);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(Rq(i, j), Ra(i, j), 1e-12);
        }
    }
}

static void test_q2mat_orthonormal() {
    vect3 rv(0.3, -0.7, 1.1);
    mat3 R = rv2m(rv);
    mat3 RtR = (~R) * R;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(RtR(i, j), (i == j) ? 1.0 : 0.0, 1e-13);
        }
    }
    EXPECT_NEAR(det(R), 1.0, 1e-13);
}

static void test_m2rv_rv2m_roundtrip() {
    vect3 rv(0.1, 0.2, -0.3);
    mat3 R = rv2m(rv);
    vect3 rv2 = m2rv(R);
    EXPECT_NEAR(rv2.i, rv.i, 1e-12);
    EXPECT_NEAR(rv2.j, rv.j, 1e-12);
    EXPECT_NEAR(rv2.k, rv.k, 1e-12);
}

static void test_qq2phi() {
    /* qq2phi(qcalcu, qreal) = q2rv(qreal * ~qcalcu) */
    quat qc = a2qua(vect3(0.1, 0.0, 0.0));
    quat qr = a2qua(vect3(0.15, 0.0, 0.0));
    vect3 phi = qq2phi(qc, qr);
    EXPECT_NEAR(phi.i, 0.05, 1e-12);
    EXPECT_NEAR(phi.j, 0.0, 1e-14);
    EXPECT_NEAR(phi.k, 0.0, 1e-14);
}

static void test_addmu() {
    quat q = a2qua(vect3(0.1, 0.2, 0.3));
    vect3 mu(0.01, -0.02, 0.03);
    quat q2 = addmu(q, mu);
    quat qexp = q * rv2q(mu);
    EXPECT_NEAR(q2.q0, qexp.q0, 1e-14);
    EXPECT_NEAR(q2.q1, qexp.q1, 1e-14);
    EXPECT_NEAR(q2.q2, qexp.q2, 1e-14);
    EXPECT_NEAR(q2.q3, qexp.q3, 1e-14);
}

static void test_UpDown() {
    /* UpDown 把姿态 [p, r, y] -> [-p, r+π, y] */
    vect3 att(0.1, 0.2, 0.3);
    quat q = a2qua(att);
    quat q2 = UpDown(q);
    vect3 att2 = q2att(q2);
    EXPECT_NEAR(att2.i, -0.1, 1e-12);
    /* roll 映射到 π - roll 或 roll + π，看约定 */
    /* 只检查 pitch 与 yaw */
    EXPECT_NEAR(att2.k, 0.3, 1e-12);
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
 * vect
 * ==========================================================================*/

static void test_vect_constructors() {
    vect v0;
    EXPECT_EQ(v0.row, 0);
    EXPECT_EQ(v0.clm, 0);
    EXPECT_EQ(v0.rc, 0);
    EXPECT_TRUE(v0.dd != nullptr);

    vect v1(5);
    EXPECT_EQ(v1.row, 5);
    EXPECT_EQ(v1.clm, 1);
    EXPECT_EQ(v1.rc, 5);
    for (int i = 0; i < 5; i++) EXPECT_NEAR(v1.dd[i], 0.0, 1e-15);

    vect v2(5, 2.5);
    EXPECT_NEAR(v2.dd[0], 2.5, 1e-15);
    EXPECT_NEAR(v2.dd[4], 2.5, 1e-15);

    vect v3(vect3(1.0, 2.0, 3.0));
    EXPECT_EQ(v3.rc, 3);
    EXPECT_NEAR(v3.dd[0], 1.0, 1e-15);
    EXPECT_NEAR(v3.dd[2], 3.0, 1e-15);

    vect v4(vect3(1, 2, 3), vect3(4, 5, 6));
    EXPECT_EQ(v4.rc, 6);
    EXPECT_NEAR(v4.dd[0], 1.0, 1e-15);
    EXPECT_NEAR(v4.dd[5], 6.0, 1e-15);
}

static void test_vect_copy_assign() {
    vect a(3, 1.0);
    a.dd[0] = 10.0;
    a.dd[1] = 20.0;
    a.dd[2] = 30.0;

    vect b = a;
    EXPECT_EQ(b.rc, 3);
    EXPECT_NEAR(b.dd[0], 10.0, 1e-15);
    EXPECT_NEAR(b.dd[2], 30.0, 1e-15);

    /* 修改 b 不影响 a（深拷贝） */
    b.dd[0] = 99.0;
    EXPECT_NEAR(a.dd[0], 10.0, 1e-15);

    /* 自赋值 */
    a = a;
    EXPECT_NEAR(a.dd[0], 10.0, 1e-15);
}

static void test_vect_arithmetic() {
    vect a(3, 1.0);
    a.dd[0] = 1; a.dd[1] = 2; a.dd[2] = 3;
    vect b(3, 1.0);
    b.dd[0] = 4; b.dd[1] = 5; b.dd[2] = 6;

    vect s = a + b;
    EXPECT_NEAR(s.dd[0], 5.0, 1e-14);
    EXPECT_NEAR(s.dd[1], 7.0, 1e-14);
    EXPECT_NEAR(s.dd[2], 9.0, 1e-14);

    vect d = a - b;
    EXPECT_NEAR(d.dd[0], -3.0, 1e-14);
    EXPECT_NEAR(d.dd[2], -3.0, 1e-14);

    vect p = a * 2.0;
    EXPECT_NEAR(p.dd[0], 2.0, 1e-14);
    EXPECT_NEAR(p.dd[2], 6.0, 1e-14);

    a += b;
    EXPECT_NEAR(a.dd[0], 5.0, 1e-14);

    a -= b;
    EXPECT_NEAR(a.dd[0], 1.0, 1e-14);

    a *= 2.0;
    EXPECT_NEAR(a.dd[0], 2.0, 1e-14);
}

static void test_vect_outer_product() {
    vect a(3, 1.0);
    a.dd[0] = 1; a.dd[1] = 2; a.dd[2] = 3;
    /* 列向量 */
    vect b = ~a;   /* 1x3 */
    EXPECT_EQ(b.row, 1);
    EXPECT_EQ(b.clm, 3);

    /* 列 × 行 -> 3x3 */
    mat M = a * b;
    EXPECT_EQ(M.row, 3);
    EXPECT_EQ(M.clm, 3);
    EXPECT_NEAR(M.dd[0], 1.0, 1e-14);
    EXPECT_NEAR(M.dd[1], 2.0, 1e-14);
    EXPECT_NEAR(M.dd[3], 2.0, 1e-14);
    EXPECT_NEAR(M.dd[4], 4.0, 1e-14);
    EXPECT_NEAR(M.dd[8], 9.0, 1e-14);
}

static void test_vect_vect3_access() {
    vect v(9, 0.0);
    v.SetVect3(0, vect3(1, 2, 3));
    v.SetVect3(3, vect3(4, 5, 6));
    v.SetVect3(6, vect3(7, 8, 9));

    vect3 g = v.GetVect3(3);
    EXPECT_NEAR(g.i, 4.0, 1e-15);
    EXPECT_NEAR(g.j, 5.0, 1e-15);
    EXPECT_NEAR(g.k, 6.0, 1e-15);

    v.Set2Vect3(0, vect3(2, 3, 4));
    EXPECT_NEAR(v.dd[0], 4.0, 1e-15);
    EXPECT_NEAR(v.dd[1], 9.0, 1e-15);
    EXPECT_NEAR(v.dd[2], 16.0, 1e-15);
}

static void test_vect_norms() {
    vect v(3, 0.0);
    v.dd[0] = 3; v.dd[1] = -4; v.dd[2] = 12;
    EXPECT_NEAR(norm(v), 13.0, 1e-13);
    EXPECT_NEAR(norm1(v), 19.0, 1e-13);
    EXPECT_NEAR(normInf(v), 12.0, 1e-13);
}

static void test_vect_mat3_assign() {
    vect v;
    mat3 M(1, 2, 3, 4, 5, 6, 7, 8, 9);
    v = M;
    EXPECT_EQ(v.rc, 9);
    EXPECT_NEAR(v.dd[0], 1.0, 1e-15);
    EXPECT_NEAR(v.dd[4], 5.0, 1e-15);
    EXPECT_NEAR(v.dd[8], 9.0, 1e-15);
}

/* ============================================================================
 * mat
 * ==========================================================================*/

static void test_mat_constructors() {
    mat m0;
    EXPECT_EQ(m0.row, 0);
    EXPECT_EQ(m0.clm, 0);
    EXPECT_EQ(m0.rc, 0);
    EXPECT_TRUE(m0.dd != nullptr);

    mat m1(3, 4);
    EXPECT_EQ(m1.row, 3);
    EXPECT_EQ(m1.clm, 4);
    EXPECT_EQ(m1.rc, 12);

    mat m2(2, 3, 7.0);
    EXPECT_NEAR(m2.dd[0], 7.0, 1e-15);
    EXPECT_NEAR(m2.dd[5], 7.0, 1e-15);
}

static void test_mat_multiplication() {
    mat A(2, 3, 0.0);
    mat B(3, 2, 0.0);
    for (int i = 0; i < 6; i++) A.dd[i] = i + 1.0;
    for (int i = 0; i < 6; i++) B.dd[i] = i + 1.0;

    mat C = A * B;
    EXPECT_EQ(C.row, 2);
    EXPECT_EQ(C.clm, 2);
    /* A = [1 2 3; 4 5 6]
       B = [1 2; 3 4; 5 6]
       C = [22 28; 49 64] */
    EXPECT_NEAR(C.dd[0], 22.0, 1e-13);
    EXPECT_NEAR(C.dd[1], 28.0, 1e-13);
    EXPECT_NEAR(C.dd[2], 49.0, 1e-13);
    EXPECT_NEAR(C.dd[3], 64.0, 1e-13);
}

static void test_mat_vect_multiplication() {
    mat A(2, 3, 0.0);
    for (int i = 0; i < 6; i++) A.dd[i] = i + 1.0;
    vect v(3, 0.0);
    v.dd[0] = 1; v.dd[1] = 1; v.dd[2] = 1;

    vect r = A * v;
    EXPECT_EQ(r.rc, 2);
    EXPECT_NEAR(r.dd[0], 6.0, 1e-13);
    EXPECT_NEAR(r.dd[1], 15.0, 1e-13);
}

static void test_mat_transpose() {
    mat A(2, 3, 0.0);
    for (int i = 0; i < 6; i++) A.dd[i] = i + 1.0;

    mat T = ~A;
    EXPECT_EQ(T.row, 3);
    EXPECT_EQ(T.clm, 2);
    EXPECT_NEAR(T.dd[0], 1.0, 1e-15);
    EXPECT_NEAR(T.dd[1], 4.0, 1e-15);
    EXPECT_NEAR(T.dd[2], 2.0, 1e-15);
    EXPECT_NEAR(T.dd[3], 5.0, 1e-15);
}

static void test_mat_diag_trace() {
    mat A(3, 3, 0.0);
    for (int i = 0; i < 9; i++) A.dd[i] = i + 1.0;
    EXPECT_NEAR(trace(A), 1 + 5 + 9, 1e-14);

    vect d = diag(A);
    EXPECT_EQ(d.rc, 3);
    EXPECT_NEAR(d.dd[0], 1.0, 1e-15);
    EXPECT_NEAR(d.dd[1], 5.0, 1e-15);
    EXPECT_NEAR(d.dd[2], 9.0, 1e-15);

    mat3 D = diag(vect3(2, 4, 6));
    EXPECT_NEAR(D.e00, 2.0, 1e-15);
    EXPECT_NEAR(D.e11, 4.0, 1e-15);
    EXPECT_NEAR(D.e22, 6.0, 1e-15);
    EXPECT_NEAR(D.e01, 0.0, 1e-15);
}

static void test_mat_eye_and_inv4() {
    mat I = eye(4);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_NEAR(I.dd[i * 4 + j], (i == j) ? 1.0 : 0.0, 1e-15);
        }
    }

    /* 4x4 对角逆 */
    mat A(4, 4, 0.0);
    for (int i = 0; i < 4; i++) A.dd[i * 4 + i] = i + 1.0;
    mat Ai = inv4(A);
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(Ai.dd[i * 4 + i], 1.0 / (i + 1.0), 1e-13);
    }
}

static void test_mat_row_col() {
    mat A(3, 3, 0.0);
    for (int i = 0; i < 9; i++) A.dd[i] = i + 1.0;

    vect r1 = A.GetRow(1);
    EXPECT_NEAR(r1.dd[0], 4.0, 1e-15);
    EXPECT_NEAR(r1.dd[1], 5.0, 1e-15);
    EXPECT_NEAR(r1.dd[2], 6.0, 1e-15);

    vect c1 = A.GetClm(1);
    EXPECT_NEAR(c1.dd[0], 2.0, 1e-15);
    EXPECT_NEAR(c1.dd[1], 5.0, 1e-15);
    EXPECT_NEAR(c1.dd[2], 8.0, 1e-15);

    A.SetRow(0, vect3(10, 20, 30));
    EXPECT_NEAR(A.dd[0], 10.0, 1e-15);
    EXPECT_NEAR(A.dd[2], 30.0, 1e-15);

    A.SetClm(0, vect3(100, 200, 300));
    EXPECT_NEAR(A.dd[0], 100.0, 1e-15);
    EXPECT_NEAR(A.dd[3], 200.0, 1e-15);
    EXPECT_NEAR(A.dd[6], 300.0, 1e-15);
}

static void test_mat_set_get_mat3() {
    mat A(6, 6, 0.0);
    mat3 M(1, 2, 3, 4, 5, 6, 7, 8, 9);

    A.SetMat3(2, 2, M);
    mat3 G = A.GetMat3(2, 2);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(G(i, j), M(i, j), 1e-15);
        }
    }
}

static void test_mat_set_askew() {
    mat A(3, 3, 0.0);
    vect3 v(1, 2, 3);
    A.SetAskew(0, 0, v);

    /* 检查反对称性 */
    EXPECT_NEAR(A.dd[0], 0.0, 1e-15);
    EXPECT_NEAR(A.dd[1], -3.0, 1e-15);
    EXPECT_NEAR(A.dd[2], 2.0, 1e-15);
    EXPECT_NEAR(A.dd[3], 3.0, 1e-15);
    EXPECT_NEAR(A.dd[4], 0.0, 1e-15);
    EXPECT_NEAR(A.dd[5], -1.0, 1e-15);
    EXPECT_NEAR(A.dd[6], -2.0, 1e-15);
    EXPECT_NEAR(A.dd[7], 1.0, 1e-15);
    EXPECT_NEAR(A.dd[8], 0.0, 1e-15);
}

static void test_mat_diag_vect3() {
    mat A(6, 6, 0.0);
    A.SetDiagVect3(1, 1, vect3(1, 2, 3));
    vect3 d = A.GetDiagVect3(1, 1);
    EXPECT_NEAR(d.i, 1.0, 1e-15);
    EXPECT_NEAR(d.j, 2.0, 1e-15);
    EXPECT_NEAR(d.k, 3.0, 1e-15);
}

/* ============================================================================
 * 坐标系
 * ==========================================================================*/

static void test_blh_xyz_roundtrip() {
    vect3 blh(30.0 * DEG, 120.0 * DEG, 100.0);
    vect3 xyz = blh2xyz(blh);
    vect3 blh2 = xyz2blh(xyz);
    EXPECT_NEAR(blh2.i, blh.i, 1e-10);
    EXPECT_NEAR(blh2.j, blh.j, 1e-10);
    EXPECT_NEAR(blh2.k, blh.k, 1e-4);
}

static void test_pos2Cen_orthonormal() {
    vect3 pos(30.0 * DEG, 120.0 * DEG, 0.0);
    mat3 C = pos2Cen(pos);
    mat3 CtC = (~C) * C;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(CtC(i, j), (i == j) ? 1.0 : 0.0, 1e-12);
        }
    }
    EXPECT_NEAR(det(C), 1.0, 1e-12);
}

static void test_Vxyz2enu() {
    /* 在赤道本初子午线：XYZ 与 ENU 对齐（U = Z） */
    vect3 pos(0.0, 0.0, 0.0);
    vect3 Vxyz(1.0, 2.0, 3.0);
    vect3 enu = Vxyz2enu(Vxyz, pos);
    /* 只断言模长不变 */
    EXPECT_NEAR(norm(enu), norm(Vxyz), 1e-12);
}

/* ============================================================================
 * 线性代数（Eigen 封装）
 * ==========================================================================*/

static void test_llt_L() {
    mat A(3, 3, 0.0);
    A.dd[0] = 4; A.dd[4] = 9; A.dd[8] = 16;
    mat L = llt_L(A);
    EXPECT_EQ(L.row, 3);
    EXPECT_NEAR(L(0, 0), 2.0, 1e-13);
    EXPECT_NEAR(L(1, 1), 3.0, 1e-13);
    EXPECT_NEAR(L(2, 2), 4.0, 1e-13);
    /* 非对角为 0 */
    EXPECT_NEAR(L(0, 1), 0.0, 1e-15);
    EXPECT_NEAR(L(1, 0), 0.0, 1e-15);

    /* 一般 SPD */
    mat B(3, 3, 0.0);
    B.dd[0] = 4; B.dd[1] = 2; B.dd[2] = 1;
    B.dd[3] = 2; B.dd[4] = 5; B.dd[5] = 3;
    B.dd[6] = 1; B.dd[7] = 3; B.dd[8] = 6;
    mat Lb = llt_L(B);
    mat LbLbt = Lb * (~Lb);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(LbLbt(i, j), B(i, j), 1e-12);
        }
    }
}

static void test_llt_inv() {
    mat A(3, 3, 0.0);
    A.dd[0] = 4; A.dd[1] = 2; A.dd[2] = 1;
    A.dd[3] = 2; A.dd[4] = 5; A.dd[5] = 3;
    A.dd[6] = 1; A.dd[7] = 3; A.dd[8] = 6;

    mat Ai = llt_inv(A);
    mat P = A * Ai;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(P(i, j), (i == j) ? 1.0 : 0.0, 1e-12);
        }
    }
}

static void test_llt_sqrtinv() {
    mat A(3, 3, 0.0);
    A.dd[0] = 4; A.dd[1] = 2; A.dd[2] = 1;
    A.dd[3] = 2; A.dd[4] = 5; A.dd[5] = 3;
    A.dd[6] = 1; A.dd[7] = 3; A.dd[8] = 6;

    mat S = llt_sqrtinv(A);
    /* S^T S = A^{-1} */
    mat StS = (~S) * S;
    mat Ai = llt_inv(A);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NEAR(StS(i, j), Ai(i, j), 1e-11);
        }
    }
}

static void test_pinv_sym() {
    /* 秩亏矩阵 */
    mat A(3, 3, 0.0);
    A.dd[0] = 2; A.dd[4] = 3;
    /* A = diag(2, 3, 0) */
    mat Ai = pinv_sym(A);
    EXPECT_NEAR(Ai(0, 0), 0.5, 1e-13);
    EXPECT_NEAR(Ai(1, 1), 1.0 / 3.0, 1e-13);
    EXPECT_NEAR(Ai(2, 2), 0.0, 1e-13);

    /* 满秩时与 inv 相同 */
    mat B(3, 3, 0.0);
    B.dd[0] = 4; B.dd[4] = 5; B.dd[8] = 6;
    mat Bi = pinv_sym(B);
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
        return vect3(A * std::cos(w), A * std::sin(w), 0.0);
    };

    vect3 pwm[2];
    pwm[0] = omega(dt * 0.25) * (dt * 0.5);
    pwm[1] = omega(dt * 0.75) * (dt * 0.5);

    /* 两个子样不平行 */
    vect3 cross01 = pwm[0] * pwm[1];
    EXPECT_TRUE(norm(cross01) > 1e-9);

    vect3 pvm[2] = {vect3(0.0), vect3(0.0)};
    imu.Update(pwm, pvm, 2, dt * 0.5);

    /* phim = (pwm[0] + pwm[1]) + (2/3)*pwm[0] × pwm[1] */
    vect3 wmm = pwm[0] + pwm[1];
    vect3 cm  = pwm[0] * (2.0 / 3.0);
    vect3 phim_expect = wmm + (cm * pwm[1]);

    EXPECT_NEAR(imu.phim.i, phim_expect.i, 1e-15);
    EXPECT_NEAR(imu.phim.j, phim_expect.j, 1e-15);
    EXPECT_NEAR(imu.phim.k, phim_expect.k, 1e-15);

    /* 补偿项非零 */
    EXPECT_TRUE(norm(cm * pwm[1]) > 1e-10);
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

    vect3 sR(1.0, 2.0, 3.0), tau(1.0, 1.0, 1.0);
    vect3 q = MKQt(sR, tau);
    EXPECT_NEAR(q.i, 2.0, 1e-14);
    EXPECT_NEAR(q.j, 8.0, 1e-14);
    EXPECT_NEAR(q.k, 18.0, 1e-14);
}

/* ============================================================================
 * main
 * ==========================================================================*/
int main() {
    std::printf("========== 数学模块单元测试 ==========\n");

    RUN_TEST(test_vect3_constructors);
    RUN_TEST(test_vect3_eigen_roundtrip);
    RUN_TEST(test_vect3_arithmetic);
    RUN_TEST(test_vect3_cross);
    RUN_TEST(test_vect3_compound_assign);
    RUN_TEST(test_vect3_queries);
    RUN_TEST(test_vect3_norms);
    RUN_TEST(test_vect3_dot_and_products);
    RUN_TEST(test_vect3_elementwise);

    RUN_TEST(test_mat3_constructors);
    RUN_TEST(test_mat3_rows_columns);
    RUN_TEST(test_mat3_arithmetic);
    RUN_TEST(test_mat3_multiplication);
    RUN_TEST(test_mat3_trace_det_inv);
    RUN_TEST(test_mat3_askew);
    RUN_TEST(test_mat3_rot);
    RUN_TEST(test_mat3_diag);
    RUN_TEST(test_mat3_dotmul_and_norm);
    RUN_TEST(test_mat3_pow);

    RUN_TEST(test_quat_constructors);
    RUN_TEST(test_quat_eigen_roundtrip);
    RUN_TEST(test_quat_multiplication);
    RUN_TEST(test_quat_inverse);
    RUN_TEST(test_quat_rotate_vector);
    RUN_TEST(test_quat_compound_assign);
    RUN_TEST(test_quat_setYaw);

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

    RUN_TEST(test_vect_constructors);
    RUN_TEST(test_vect_copy_assign);
    RUN_TEST(test_vect_arithmetic);
    RUN_TEST(test_vect_outer_product);
    RUN_TEST(test_vect_vect3_access);
    RUN_TEST(test_vect_norms);
    RUN_TEST(test_vect_mat3_assign);

    RUN_TEST(test_mat_constructors);
    RUN_TEST(test_mat_multiplication);
    RUN_TEST(test_mat_vect_multiplication);
    RUN_TEST(test_mat_transpose);
    RUN_TEST(test_mat_diag_trace);
    RUN_TEST(test_mat_eye_and_inv4);
    RUN_TEST(test_mat_row_col);
    RUN_TEST(test_mat_set_get_mat3);
    RUN_TEST(test_mat_set_askew);
    RUN_TEST(test_mat_diag_vect3);

    RUN_TEST(test_blh_xyz_roundtrip);
    RUN_TEST(test_pos2Cen_orthonormal);
    RUN_TEST(test_Vxyz2enu);

    RUN_TEST(test_llt_L);
    RUN_TEST(test_llt_inv);
    RUN_TEST(test_llt_sqrtinv);
    RUN_TEST(test_pinv_sym);

    RUN_TEST(test_imu_coning_compensation);
    RUN_TEST(test_scalar_utils);
    RUN_TEST(test_swapt);
    RUN_TEST(test_norm_ptrs);
    RUN_TEST(test_MKQt);

    std::printf("=============================================\n");
    std::printf("总断言: %d, 通过: %d, 失败: %d\n",
                g_pass + g_fail, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}