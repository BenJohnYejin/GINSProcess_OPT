# 第 X 章 GNSS/INS 滑窗图优化松组合导航理论

本章建立 GNSS/INS 松组合滑窗图优化的完整理论框架。首先定义坐标系与状态量；其次推导 IMU 预积分的连续与离散形式及其误差传播；然后在流形上定义状态扰动与切空间运算；随后推导各观测因子（GNSS 位置、速度、航向、静止零速、里程计速度）的残差模型与雅可比；最后给出边缘化的 Schur 补推导与整体优化问题的求解。

---

## X.1 坐标系与符号约定

### X.1.1 坐标系定义

定义如下三个正交坐标系：

- **惯性系 $i$**：地心惯性坐标系。
- **地球系 $e$**：地心地固坐标系。
- **导航系 $n$**：东-北-天（ENU）局部水平坐标系，原点取第一个有效 GNSS 观测点。
- **机体系 $b$**：右-前-上（RFU），原点取 IMU 中心。
- **里程计系 $m$**：里程计安装方向，理想情况下 $x_m$ 沿前进方向。

### X.1.2 符号约定

- 四元数 $q=[q_0,q_1,q_2,q_3]=[w,x,y,z]$ 表示 $b\to n$ 旋转，旋转矩阵记为 $R(q)\in SO(3)$。
- 欧拉角 $\text{att}=[\text{pitch},\text{roll},\text{yaw}]$，姿态解算约定为

$$
\text{pitch}=\arcsin(R_{21}),\quad
\text{roll}=\operatorname{atan2}(-R_{20},R_{22}),\quad
\text{yaw}=\operatorname{atan2}(-R_{01},R_{11})
\tag{X.1}
$$

- 向量叉乘矩阵（反对称矩阵）：

$$
[a]_\times=\begin{bmatrix}
0&-a_z&a_y\\
a_z&0&-a_x\\
-a_y&a_x&0
\end{bmatrix}
\tag{X.2}
$$

- 四元数指数映射与对数映射：

$$
\mathrm{Exp}(\phi)=\begin{bmatrix}\cos(\|\phi\|/2)\\[2pt]\dfrac{\phi}{\|\phi\|}\sin(\|\phi\|/2)\end{bmatrix},\qquad
\mathrm{Log}(q)=2\arccos(q_0)\frac{q_v}{\|q_v\|}
\tag{X.3}
$$

### X.1.3 状态量定义

滑窗内第 $k$ 个关键帧的状态定义为

$$
x_k=(p_k,q_k,v_k,b_{g,k},b_{a,k},s_{\text{odo},k})
\tag{X.4}
$$

其中 $p_k\in\mathbb R^3$ 为局部 ENU 位置，$q_k\in S^3$ 为 $b\to n$ 姿态，$v_k\in\mathbb R^3$ 为 ENU 速度，$b_{g,k},b_{a,k}\in\mathbb R^3$ 为陀螺与加表零偏，$s_{\text{odo},k}\in\mathbb R$ 为里程计刻度因子误差。整体状态 $x=\{x_k\}_{k=0}^{N-1}$，其中 $N$ 为滑窗内关键帧数。

---

## X.2 IMU 预积分理论

### X.2.1 连续时间运动学

IMU 的陀螺与加表测量模型为

$$
\omega_m=\omega+b_g+n_g,\qquad a_m=a+b_a+n_a
\tag{X.5}
$$

其中 $\omega,a$ 为真实角速度与比力，$n_g,n_a$ 为白噪声。导航方程在 $n$ 系下为

$$
\dot p=v,\qquad
\dot v=R(q)a+g,\qquad
\dot q=\tfrac12 q\otimes(\omega_m-b_g-n_g)
\tag{X.6}
$$

零偏随机游走：

$$
\dot b_g=n_{bg},\qquad \dot b_a=n_{ba}
\tag{X.7}
$$

**注 X.1** 式 (X.6) 中 $\dot q$ 的右乘形式与四元数约定一致：$q\otimes\delta q$ 表示在机体系施加扰动。

### X.2.2 预积分量的定义

在起始帧 $b_i$ 系中定义预积分量

$$
\Delta p_{ij}=R_i^T\left(p_j-p_i-v_i\Delta t-\tfrac12 g\Delta t^2\right)
\tag{X.8}
$$

$$
\Delta v_{ij}=R_i^T(v_j-v_i-g\Delta t)
\tag{X.9}
$$

$$
\Delta q_{ij}=q_i^{-1}\otimes q_j
\tag{X.10}
$$

其中 $\Delta t=t_j-t_i$，$g$ 为局部 ENU 下的重力矢量（指向下）。由式 (X.6) 直接可得预积分量的微分方程

$$
\dot{\Delta p}=\Delta v,\qquad
\dot{\Delta v}=R(\Delta q)(a_m-b_a-n_a),\qquad
\dot{\Delta q}=\tfrac12\Delta q\otimes(\omega_m-b_g-n_g)
\tag{X.11}
$$

### X.2.3 中值离散积分

设第 $k$ 个 IMU 区间时长为 $\Delta t_k$，去偏后测量为

$$
\bar\omega_k=\omega_{m,k}-b_g,\qquad \bar a_k=a_{m,k}-b_a
\tag{X.12}
$$

中值积分离散形式为

$$
\Delta q_{k+1}=\Delta q_k\otimes\mathrm{Exp}(\bar\omega_k\Delta t_k)
\tag{X.13}
$$

$$
\Delta q_{\mathrm{mid}}=\Delta q_k\otimes\mathrm{Exp}(\tfrac12\bar\omega_k\Delta t_k)
\tag{X.14}
$$

$$
\Delta v_{k+1}=\Delta v_k+R(\Delta q_{\mathrm{mid}})\bar a_k\Delta t_k
\tag{X.15}
$$

$$
\Delta p_{k+1}=\Delta p_k+\Delta v_k\Delta t_k+\tfrac12R(\Delta q_{\mathrm{mid}})\bar a_k\Delta t_k^2
\tag{X.16}
$$

**定理 X.1**（中值积分精度）当 $\bar\omega_k\Delta t_k$ 与 $\bar a_k\Delta t_k$ 为小量时，式 (X.13)–(X.16) 的截断误差为 $O(\Delta t_k^3)$。

**证明** 将真实积分在区间中点展开，与式 (X.13)–(X.16) 比较，梯形公式（中点）的局部截断误差为 $O(\Delta t_k^3)$。$\square$

### X.2.4 误差状态定义

定义误差状态

$$
\Delta p=\Delta\hat p+\delta p,\quad
\Delta v=\Delta\hat v+\delta v,\quad
\Delta q=\Delta\hat q\otimes\mathrm{Exp}(\delta\phi)
\tag{X.17}
$$

$$
b_g=\hat b_g+\delta b_g,\quad
b_a=\hat b_a+\delta b_a
\tag{X.18}
$$

其中 $\delta\phi$ 为机体系下的右扰动。误差状态向量按如下顺序排列：

$$
\delta x=[\delta p,\delta v,\delta\phi,\delta b_g,\delta b_a]\in\mathbb R^{15}
\tag{X.19}
$$

### X.2.5 连续时间误差方程

对式 (X.11) 在名义值处线性化，得到误差微分方程

$$
\dot{\delta p}=\delta v
\tag{X.20}
$$

$$
\dot{\delta v}=-R(\Delta q)[a]_\times\delta\phi-R(\Delta q)\delta b_a-R(\Delta q)n_a
\tag{X.21}
$$

$$
\dot{\delta\phi}=-[\omega]_\times\delta\phi-\delta b_g-n_g
\tag{X.22}
$$

$$
\dot{\delta b_g}=n_{bg},\qquad \dot{\delta b_a}=n_{ba}
\tag{X.23}
$$

**证明** 以 $\dot{\delta v}$ 为例。由式 (X.11) 第二式，名义量为

$$
\dot{\Delta\hat v}=R(\Delta\hat q)\bar a
$$

真实量为

$$
\dot{\Delta v}=R(\Delta\hat q\otimes\mathrm{Exp}(\delta\phi))(\bar a-\delta b_a-n_a)
$$

利用 $R(\mathrm{Exp}(\delta\phi))\approx I+[\delta\phi]_\times$，得

$$
R(\Delta q)\approx R(\Delta\hat q)(I+[\delta\phi]_\times)
$$

代入并减去名义式，忽略二阶小量：

$$
\dot{\delta v}=R(\Delta\hat q)\big([\delta\phi]_\times\bar a-\delta b_a-n_a\big)
=-R(\Delta\hat q)[\bar a]_\times\delta\phi-R(\Delta\hat q)\delta b_a-R(\Delta\hat q)n_a
$$

即式 (X.21)。其余类似。$\square$

### X.2.6 离散时间误差传播

将式 (X.20)–(X.23) 离散化，得到一阶形式

$$
\delta x_{k+1}=F_k\delta x_k+G_k w_k,\qquad
w_k=[n_g,n_a,n_{bg},n_{ba}]^T
\tag{X.24}
$$

其中

$$
F_k=I+A_k\Delta t_k,\qquad
A_k=\begin{bmatrix}
0&I&0&0&0\\
0&0&-R[a]_\times&0&-R\\
0&0&-[\omega]_\times&-I&0\\
0&0&0&0&0\\
0&0&0&0&0
\end{bmatrix}
\tag{X.25}
$$

$$
G_k=\begin{bmatrix}
0&0&0&0\\
0&-R&0&0\\
-I&0&0&0\\
0&0&I&0\\
0&0&0&I
\end{bmatrix}
\tag{X.26}
$$

过程噪声协方差

$$
Q_k=\mathrm{diag}\big(\sigma_g^2\Delta t_k I,\ \sigma_a^2\Delta t_k I,\ \tfrac{2\sigma_{bg}^2}{\tau}\Delta t_k I,\ \tfrac{2\sigma_{ba}^2}{\tau}\Delta t_k I\big)
\tag{X.27}
$$

协方差传播与状态转移矩阵累积：

$$
P_{k+1}=F_kP_kF_k^T+G_kQ_kG_k^T,\qquad
\Phi_{k+1}=F_k\Phi_k
\tag{X.28}
$$

其中 $\Phi_k=\partial\delta x_k/\partial\delta x_i$。

### X.2.7 零偏一阶补偿

当零偏估计偏离线性化点时，预积分量可用一阶展开补偿。设

$$
\delta b_g=b_{g,i}-b_g^{lin},\qquad \delta b_a=b_{a,i}-b_a^{lin}
\tag{X.29}
$$

则

$$
\Delta p_{\mathrm{corr}}=\Delta p+J_{p,b_g}\delta b_g+J_{p,b_a}\delta b_a
\tag{X.30}
$$

$$
\Delta v_{\mathrm{corr}}=\Delta v+J_{v,b_g}\delta b_g+J_{v,b_a}\delta b_a
\tag{X.31}
$$

$$
\Delta q_{\mathrm{corr}}=\Delta q\otimes\mathrm{Exp}(J_{q,b_g}\delta b_g)
\tag{X.32}
$$

其中 $J_{\cdot,b}=\Phi$ 的对应分块：

$$
J_{p,b_g}=\Phi_{0,9},\ J_{p,b_a}=\Phi_{0,12},\ J_{v,b_g}=\Phi_{3,9},\ J_{v,b_a}=\Phi_{3,12},\ J_{q,b_g}=\Phi_{6,9}
\tag{X.33}
$$

### X.2.8 预积分残差

定义残差向量 $r=[r_p,r_v,r_\phi,r_{bg},r_{ba}]\in\mathbb R^{15}$：

$$
r_p=R_i^T\left(p_j-p_i-v_i\Delta t-\tfrac12g\Delta t^2\right)-\Delta p_{\mathrm{corr}}
\tag{X.34}
$$

$$
r_v=R_i^T(v_j-v_i-g\Delta t)-\Delta v_{\mathrm{corr}}
\tag{X.35}
$$

$$
r_\phi=2\left[\Delta q_{\mathrm{corr}}^{-1}\otimes q_i^{-1}\otimes q_j\right]_{\mathrm{vec}}
\tag{X.36}
$$

$$
r_{bg}=b_{g,j}-b_{g,i},\qquad r_{ba}=b_{a,j}-b_{a,i}
\tag{X.37}
$$

若按预积分协方差白化，则乘 $S=P_{ij}^{-1/2}$，得 $r\leftarrow Sr$。

---

## X.3 状态流形与切空间

### X.3.1 流形的定义

状态 $x$ 所在流形为

$$
\mathcal M=\mathbb R^3\times S^3\times\mathbb R^3\times\mathbb R^3\times\mathbb R^3\ (\times\mathbb R)
\tag{X.38}
$$

其在单位元处的切空间维数为

$$
\dim\mathcal T_x\mathcal M=3+3+3+3+3=15\quad(\text{含里程计为 }16)
\tag{X.39}
$$

### X.3.2 右扰动与切空间运算

在 $q$ 处定义右扰动：

$$
q'=q\otimes\mathrm{Exp}(\delta\phi)
\tag{X.40}
$$

对应旋转矩阵近似：

$$
R(q')\approx R(q)(I+[\delta\phi]_\times)
\tag{X.41}
$$

流形加法 $\boxplus$ 与减法 $\boxminus$ 定义为

$$
\text{plus}(s,\delta):\quad
\begin{cases}
p'=p+\delta p\\
q'=q\otimes\mathrm{Exp}(\delta\phi)\\
v'=v+\delta v\\
b_g'=b_g+\delta b_g\\
b_a'=b_a+\delta b_a
\end{cases}
\tag{X.42}
$$

$$
\text{minus}(a,b):\quad
\begin{cases}
\delta p=a.p-b.p\\
\delta\phi=\mathrm{Log}(b.q^{-1}\otimes a.q)\\
\delta v=a.v-b.v\\
\delta b_g=a.b_g-b.b_g\\
\delta b_a=a.b_a-b.b_a
\end{cases}
\tag{X.43}
$$

**命题 X.1**（对偶性）对任意 $a,b\in\mathcal M$，有

$$
\text{plus}(b,\text{minus}(a,b))=a
\tag{X.44}
$$

**证明** 对旋转分量：

$$
b.q\otimes\mathrm{Exp}\big(\mathrm{Log}(b.q^{-1}\otimes a.q)\big)=b.q\otimes b.q^{-1}\otimes a.q=a.q
$$

其余分量由加法/减法互逆即得。$\square$

### X.3.3 Ceres 流形雅可比

在 Ceres 中，`PoseManifold` 的环境维数为 7，切空间维数为 6。`PlusJacobian` 为 $7\times 6$：

$$
J_{\mathrm{plus}}=\begin{bmatrix}
I_3&0\\
0&\dfrac{\partial q}{\partial\delta\phi}
\end{bmatrix}
\tag{X.45}
$$

其中

$$
\frac{\partial q}{\partial\delta\phi}=\frac12
\begin{bmatrix}
-q_x&-q_y&-q_z\\
q_w&-q_z&q_y\\
q_z&q_w&-q_x\\
-q_y&q_x&q_w
\end{bmatrix}
\tag{X.46}
$$

`MinusJacobian` 为 $6\times 7$：

$$
J_{\mathrm{minus}}=\begin{bmatrix}
I_3&0\\
0&J_q
\end{bmatrix},\qquad
J_q=2\begin{bmatrix}
-q_y&q_x&-q_w&q_z\\
-q_z&q_w&q_x&-q_y\\
-q_w&-q_z&q_y&q_x
\end{bmatrix}
\tag{X.47}
$$

---

## X.4 观测因子建模

### X.4.1 GNSS 位置因子

观测模型：

$$
p_{\text{gnss}}=p+n_p,\qquad n_p\sim\mathcal N(0,\Sigma_p)
\tag{X.48}
$$

残差：

$$
r_p=\Sigma_p^{-1/2}(p-p_{\text{gnss}})
\tag{X.49}
$$

参数块：pose(7)。残差维数 3。对 $p$ 的雅可比为 $\Sigma_p^{-1/2}$，对 $q,v,b_g,b_a$ 为 0。

### X.4.2 GNSS 速度因子

观测模型：

$$
v_{\text{gnss}}=v+n_v,\qquad n_v\sim\mathcal N(0,\Sigma_v)
\tag{X.50}
$$

残差：

$$
r_v=\Sigma_v^{-1/2}(v-v_{\text{gnss}})
\tag{X.51}
$$

参数块：mix(10)。残差维数 3。对 $v$ 的雅可比为 $\Sigma_v^{-1/2}$，对其余分量为 0。加入 Huber 核可抑制野值。

### X.4.3 GNSS 航向因子

观测模型：

$$
\psi_{\text{gnss}}=\text{yaw}(q)+n_\psi,\qquad n_\psi\sim\mathcal N(0,\sigma_\psi^2)
\tag{X.52}
$$

其中 $\text{yaw}(q)$ 由式 (X.1) 给出：

$$
\text{yaw}(q)=\operatorname{atan2}(-R_{01},R_{11})
=\operatorname{atan2}\big(2(q_0q_3-q_1q_2),\ q_0^2-q_1^2+q_2^2-q_3^2\big)
\tag{X.53}
$$

残差：

$$
r_\psi=\frac{1}{\sigma_\psi}\operatorname{wrap}\big(\text{yaw}(q)-\psi_{\text{gnss}}\big)
\tag{X.54}
$$

其中角度规整

$$
\operatorname{wrap}(d)=\operatorname{atan2}(\sin d,\cos d)
\tag{X.55}
$$

参数块：pose(7)。残差维数 1。

**定理 X.2**（航向对右扰动的雅可比）在右扰动 $q'=q\otimes\mathrm{Exp}(\delta\phi)$ 下，航向对 $\delta\phi$ 的雅可比为

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=\frac{1}{R_{01}^2+R_{11}^2}
\big(-R_{11}R_{02}+R_{01}R_{12},\ 0,\ R_{11}R_{00}-R_{01}R_{10}\big)
\tag{X.56}
$$

进一步可化简为

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=\left(-\frac{\tan(\text{roll})}{\cos(\text{pitch})},\ 0,\ \frac{\cos(\text{roll})}{\cos(\text{pitch})}\right)
\tag{X.57}
$$

**证明** 记 $N=-R_{01}$，$D=R_{11}$，则 $\text{yaw}=\operatorname{atan2}(N,D)$。由链式法则

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=\frac{D\frac{\partial N}{\partial\delta\phi}-N\frac{\partial D}{\partial\delta\phi}}{N^2+D^2}
$$

在右扰动下，$R'=R(I+[\delta\phi]_\times)$，故

$$
\frac{\partial R(i,1)}{\partial\delta\phi}=\big(R(i,2),0,-R(i,0)\big)
$$

代入 $N,D$ 得

$$
\frac{\partial N}{\partial\delta\phi}=(-R_{02},0,R_{00}),\qquad
\frac{\partial D}{\partial\delta\phi}=(R_{12},0,-R_{10})
$$

故

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=\frac{1}{R_{01}^2+R_{11}^2}
\big(-R_{11}R_{02}+R_{01}R_{12},\ 0,\ R_{11}R_{00}-R_{01}R_{10}\big)
$$

即式 (X.56)。再将 $R$ 的欧拉角展开式代入，$R_{01}^2+R_{11}^2=c_i^2$，第一分量

$$
-R_{11}R_{02}+R_{01}R_{12}=-c_is_j
$$

第三分量

$$
R_{11}R_{00}-R_{01}R_{10}=c_ic_j
$$

得到式 (X.57)。$\square$

**注 X.2** 式 (X.57) 中第二分量为 0，表明绕机体横轴旋转不改变航向；pitch $\to\pm90°$ 时雅可比发散，对应万向节锁。

### X.4.4 静止零速因子（ZUPT）

观测模型：

$$
v_{\text{zupt}}=0+n_v,\qquad n_v\sim\mathcal N(0,\sigma_{\text{zupt}}^2I)
\tag{X.58}
$$

残差：

$$
r_{\text{zupt}}=\frac{1}{\sigma_{\text{zupt}}}v
\tag{X.59}
$$

参数块：mix(10)。残差维数 3。触发条件：滑动窗口内陀螺与加表去均值 RMS 小于阈值：

$$
\sigma_\omega<\tau_\omega,\qquad \sigma_a<\tau_a
\tag{X.60}
$$

**注 X.3** 静止时 ZUPT 使速度被绝对约束，进而使零偏可观测。

### X.4.5 里程计速度因子

里程计测量前向位移 $dS$，速度模型为

$$
v_{\text{odo}}^b=(1+s_{\text{odo}})\frac{dS}{dt}C_b^m(\boldsymbol\alpha)\mathbf e_x+\boldsymbol\omega_{ib}^b\times\mathbf l_{\text{OD}}
\tag{X.61}
$$

其中安装角 $\boldsymbol\alpha=[\alpha_p,0,\alpha_y]^T$，且

$$
C_b^m\mathbf e_x=
\begin{bmatrix}
\cos\alpha_y\cos\alpha_p\\
-\sin\alpha_y\cos\alpha_p\\
-\sin\alpha_p
\end{bmatrix}
\tag{X.62}
$$

观测模型：

$$
v_{\text{odo}}^n=R_b^n v_{\text{odo}}^b+n_{\text{odo}},\qquad n_{\text{odo}}\sim\mathcal N(0,\sigma_{\text{odo}}^2I)
\tag{X.63}
$$

残差：

$$
r_{\text{odo}}=R_b^n v_{\text{odo}}^b-v^n
\tag{X.64}
$$

参数块：pose(7)、mix(10)。残差维数 3。刻度因子 $s_{\text{odo}}$ 位于 mix[9]，加性更新。

**注 X.4**（可观性）$s_{\text{odo}}$ 与速度耦合，$\alpha_y$ 与航向耦合，$\alpha_p$ 与俯仰耦合，需 GNSS 速度/位置/航向或双天线航向才能分离。

---

## X.5 边缘化理论

### X.5.1 法方程累积

对每个残差块 $r_k(x)$ 在当前线性化点处展开

$$
r_k(x+\delta x)\approx r_k(x)+J_k\delta x
\tag{X.65}
$$

高斯牛顿最小化 $\sum_k\|r_k+J_k\delta x\|^2$ 得到法方程

$$
H\delta x=b,\qquad H=\sum_k J_k^TJ_k,\qquad b=\sum_k J_k^Tr_k
\tag{X.66}
$$

### X.5.2 Schur 补

将状态分为待边缘化变量 $\delta x_m$ 与保留变量 $\delta x_r$：

$$
\begin{bmatrix}
H_{mm}&H_{mr}\\
H_{mr}^T&H_{rr}
\end{bmatrix}
\begin{bmatrix}
\delta x_m\\
\delta x_r
\end{bmatrix}
=
\begin{bmatrix}
b_m\\
b_r
\end{bmatrix}
\tag{X.67}
$$

由第一行消去 $\delta x_m$：

$$
H^*\delta x_r=b^*
\tag{X.68}
$$

其中

$$
H^*=H_{rr}-H_{mr}^TH_{mm}^{-1}H_{mr}
\tag{X.69}
$$

$$
b^*=b_r-H_{mr}^TH_{mm}^{-1}b_m
\tag{X.70}
$$

若 $H_{mm}$ 非正定，采用特征分解伪逆：

$$
H_{mm}^{-1}\approx V D^{-1}V^T
\tag{X.71}
$$

仅保留 $|d_i|>\epsilon\max_j|d_j|$ 的特征方向。

### X.5.3 先验恢复

将 $(H^*,b^*)$ 分解为 $H^*=J_{\text{out}}^TJ_{\text{out}}$，$b^*=J_{\text{out}}^Tr_{\text{out}}$。若 Cholesky 分解 $H^*=LL^T$，则取

$$
J_{\text{out}}=L^T,\qquad r_{\text{out}}=L^{-1}b^*
\tag{X.72}
$$

若用特征分解 $H^*=VDV^T$，则取

$$
J_{\text{out}}=D^{1/2}V^T,\qquad r_{\text{out}}=D^{-1/2}V^Tb^*
\tag{X.73}
$$

先验因子残差为

$$
r_{\text{prior}}(\delta x)=J_{\text{prior}}\delta x-r_{\text{prior}}
\tag{X.74}
$$

**命题 X.2**（先验一致性）由式 (X.72) 或 (X.73) 定义的 $(J_{\text{out}},r_{\text{out}})$ 满足

$$
J_{\text{out}}^TJ_{\text{out}}=H^*,\qquad J_{\text{out}}^Tr_{\text{out}}=b^*
\tag{X.75}
$$

**证明** 对式 (X.72)：$J_{\text{out}}^TJ_{\text{out}}=LL^T=H^*$，$J_{\text{out}}^Tr_{\text{out}}=LL^{-1}b^*=b^*$。对式 (X.73) 同理。$\square$

---

## X.6 整体优化问题与求解

### X.6.1 最大后验估计

给定所有观测 $\mathcal Z=\{z_k\}$，状态 $x$ 的最大后验估计为

$$
x^*=\arg\max_x\ p(x|\mathcal Z)=\arg\max_x\ p(x)p(\mathcal Z|x)
\tag{X.76}
$$

在高斯假设下等价于非线性最小二乘

$$
x^*=\arg\min_x\ \frac12\sum_k\|r_k(x)\|^2
\tag{X.77}
$$

其中残差按观测协方差白化。

### X.6.2 整体残差

综合各因子，整体残差为

$$
r(x)=\begin{bmatrix}
\{r_{\text{imu},k}\}_{k=0}^{N-2}\\
\{r_{\text{gnss},k}^{\text{pos}}\}_{k\in\mathcal G_p}\\
\{r_{\text{gnss},k}^{\text{vel}}\}_{k\in\mathcal G_v}\\
\{r_{\text{gnss},k}^{\text{yaw}}\}_{k\in\mathcal G_\psi}\\
\{r_{\text{zupt},k}\}_{k\in\mathcal S}\\
\{r_{\text{odo},k}\}_{k\in\mathcal O}\\
r_{\text{prior}}
\end{bmatrix}
\tag{X.78}
$$

### X.6.3 Levenberg-Marquardt 迭代

在第 $t$ 次迭代，线性化并求解

$$
(H+\lambda I)\delta x=-b
\tag{X.79}
$$

更新

$$
x^{(t+1)}=x^{(t)}\boxplus\delta x
\tag{X.80}
$$

其中 $\boxplus$ 为式 (X.42) 定义的流形加法。

### X.6.4 滑窗与边缘化

窗口满时，将最旧帧的切空间变量作为 $\delta x_m$ 消去，得到先验 $(J_{\text{out}},r_{\text{out}})$，加入下一轮优化。状态维数保持固定，计算量有界。

---

## X.7 本章小结

本章建立了 GNSS/INS 滑窗图优化松组合的完整理论：

1. **预积分**：在 $b_i$ 系中定义 $\Delta p,\Delta v,\Delta q$，推导中值离散与误差传播，得到残差 (X.34)–(X.37)；
2. **流形**：采用右扰动，定义 $\boxplus,\boxminus$ 与 Ceres 雅可比 (X.45)–(X.47)；
3. **观测因子**：推导 GNSS 位置/速度/航向、ZUPT、里程计速度的残差与雅可比，其中航向雅可比 (X.57) 具有简洁闭式；
4. **边缘化**：Schur 补 (X.69)–(X.70) 与先验恢复 (X.72)–(X.73)；
5. **整体优化**：MAP 估计 (X.76)、整体残差 (X.78)、LM 迭代 (X.79)。

该框架在流形上统一处理旋转，利用 Ceres AutoDiff 自动求导，通过滑窗与边缘化实现固定计算量的实时估计。
# 第X章 里程计与双天线航向联合在线标定理论

## X.1 引言

在GNSS/INS/里程计多源融合导航系统中，各传感器之间的安装关系是影响融合精度的关键因素。里程计的刻度因子、安装角与杆臂，以及双天线GNSS基线相对IMU机体系的航向安装偏置，均属于硬件安装决定的常量参数。这些参数若存在标定误差，将直接引入系统性偏差，降低组合导航的精度与可靠性。

传统的标定方法通常将里程计标定与双天线航向标定分离处理，或采用卡尔曼滤波框架进行序贯估计。前者忽略了各标定量之间通过共享状态变量相互耦合的信息，后者受限于单步观测，无法充分利用多帧历史信息进行联合优化。此外，现有方法中，标定过程与导航过程通常采用分离的架构，需维护两套代码分支，工程实现复杂。

本章建立滑窗图优化框架下的里程计与双天线航向联合在线标定理论。首先给出标定问题的一般描述与参数块划分准则；其次分别推导里程计速度因子与双天线航向因子的残差模型及其对各标定参数的雅可比；然后给出全局法方程累积与Levenberg-Marquardt迭代更新公式；最后分析各标定参数的可观测性条件。

## X.2 标定问题描述与参数块划分

### X.2.1 标定量的物理含义

本程序需要标定的物理量分为两类。第一类为里程计参数，包括刻度因子误差$s$、安装角$\boldsymbol\alpha=[\alpha_p,0,\alpha_y]^T$以及杆臂$\mathbf l_{\text{OD}}$。第二类为双天线航向偏置$\psi_{\text{off}}$，其定义为双天线基线方向相对IMU机体系前向轴的安装夹角。这些量具有如下共同特征：

（1）物理上不随时间变化，是硬件安装决定的常量；

（2）在所有关键帧上共享同一组取值；

（3）不出现在逐帧状态中，但仍以全局参数块的形式参与优化。

### X.2.2 状态变量与全局标定参数

定义滑窗内第$k$个关键帧的状态为

$$
x_k=(p_k,\ q_k,\ v_k,\ b_{g,k},\ b_{a,k})\in\mathbb R^3\times S^3\times\mathbb R^3\times\mathbb R^3\times\mathbb R^3
\tag{X.1}
$$

其中$p_k$为局部ENU位置，$q_k$为$b\to n$姿态四元数，$v_k$为ENU速度，$b_{g,k}$与$b_{a,k}$分别为陀螺零偏与加表零偏。

定义全局标定向量为

$$
c=(s,\ \alpha_p,\ \alpha_y,\ \mathbf l_{\text{OD}},\ \psi_{\text{off}})\in\mathbb R^7
\tag{X.2}
$$

其数据布局为

$$
c[0..6]=\big(s,\ \alpha_p,\ \alpha_y,\ l_x,\ l_y,\ l_z,\ \psi_{\text{off}}\big)
\tag{X.3}
$$

各分量的物理含义与维数如表X.1所示。

**表X.1 全局标定参数一览**

| 符号 | 含义 | 维数 |
|---|---|---|
| $s$ | 里程计刻度因子误差 | 1 |
| $\alpha_p$ | 里程计安装俯仰角 | 1 |
| $\alpha_y$ | 里程计安装偏航角 | 1 |
| $\mathbf l_{\text{OD}}$ | 里程计杆臂（机体系） | 3 |
| $\psi_{\text{off}}$ | 双天线航向偏置 | 1 |

标定参数采用加性参数化，无需流形结构，切空间维数等于环境维数7。

### X.2.3 参数块划分准则

图优化中，参数块的划分应遵循以下准则：

（1）逐帧状态用于描述随时间变化的位姿、速度与零偏；

（2）全局参数用于描述不随时间变化的硬件常量；

（3）一个参数能否被优化，取决于残差函数是否引用该参数块，以及残差对该参数块的雅可比是否非零，与该参数是否属于"状态"无关。

参数块的生命周期与边缘化处理如表X.2所示。

**表X.2 参数块划分与生命周期**

| 参数块 | 物理含义 | 生命周期 | 边缘化处理 |
|---|---|---|---|
| 逐帧状态$x_k$ | 随时间变化的量 | 帧在窗口内 | 消去 |
| 全局标定$c$ | 硬件决定的常量 | 整个会话 | 不消去 |

**命题X.1（标定参数的可优化条件）** 标定参数可作为全局参数块被优化，当且仅当存在至少一个残差函数引用了该参数块，并且残差对该参数块的雅可比非零。

**证明** 见X.5节法方程累积过程。由式(X.40)可知，若某标定参数$c_i$未出现在任何残差函数中，则$H_{cc}$的第$i$行第$i$列为零，法方程对应的右端项$b_c$的第$i$个分量也为零。此时Levenberg-Marquardt迭代的增量$\delta c_i$恒为零，参数保持初值不变。反之，若存在残差引用该参数且雅可比非零，则$H_{cc}$的第$i$个对角元非零，$\delta c_i$由法方程解出。证毕。

## X.3 里程计速度因子

### X.3.1 里程计测量模型

里程计只测量沿其安装方向$x_m$的位移增量$dS$。定义里程计坐标系$m$，其$x_m$轴沿前进方向。里程计在$m$系下的位移向量为

$$
\Delta\mathbf s^{m}=\big(dS,\ 0,\ 0\big)^T
\tag{X.4}
$$

设里程计真实位移与测量位移的关系为

$$
dS_{\text{true}}=(1+s)\,dS_{\text{meas}}
\tag{X.5}
$$

其中$s$为刻度因子误差。

### X.3.2 安装角与坐标变换

安装角$\boldsymbol\alpha=[\alpha_p,0,\alpha_y]^T$给出$m\to b$的旋转矩阵

$$
C_b^m(\boldsymbol\alpha)=
\begin{bmatrix}
\cos\alpha_y\cos\alpha_p & \sin\alpha_y & \cos\alpha_y\sin\alpha_p\\
-\sin\alpha_y\cos\alpha_p & \cos\alpha_y & -\sin\alpha_y\sin\alpha_p\\
-\sin\alpha_p & 0 & \cos\alpha_p
\end{bmatrix}
\tag{X.6}
$$

将里程计位移从$m$系转到$b$系：

$$
\Delta\mathbf s^{b}=C_b^m(\boldsymbol\alpha)\,\Delta\mathbf s^{m}=(1+s)\,dS\,\mathbf c_x(\boldsymbol\alpha)
\tag{X.7}
$$

其中定义方向单位向量

$$
\mathbf c_x(\boldsymbol\alpha)\triangleq C_b^m(\boldsymbol\alpha)\,\mathbf e_x=
\begin{bmatrix}
\cos\alpha_y\cos\alpha_p\\
-\sin\alpha_y\cos\alpha_p\\
-\sin\alpha_p
\end{bmatrix}
\tag{X.8}
$$

### X.3.3 杆臂补偿

里程计安装在机体系$\mathbf l_{\text{OD}}$处。该点的速度与IMU中心速度的关系为

$$
\mathbf v_{\text{OD}}^{b}=\mathbf v^{b}+\boldsymbol\omega_{ib}^{b}\times\mathbf l_{\text{OD}}
\tag{X.9}
$$

其中$\boldsymbol\omega_{ib}^{b}$为陀螺测量的角速度（已去零偏）。

### X.3.4 里程计速度观测方程

合并式(X.7)与(X.9)，里程计在机体系下的速度估计为

$$
\mathbf v_{\text{odo}}^{b}=(1+s)\frac{dS}{dt}\,\mathbf c_x(\boldsymbol\alpha)
+\boldsymbol\omega_{ib}^{b}\times\mathbf l_{\text{OD}}
\tag{X.10}
$$

转到导航系，得到观测方程

$$
\mathbf v_{\text{odo}}^{n}=R_b^n(q)\,\mathbf v_{\text{odo}}^{b}
\tag{X.11}
$$

其中$R_b^n(q)$为机体系到导航系的旋转矩阵。

### X.3.5 残差定义

里程计因子将式(X.11)的估计与状态速度$v^n$比较，定义标准化残差

$$
\mathbf r_{\text{odo}}=
\frac{1}{\sigma_{\text{odo}}}
\Big(R_b^n(q)\,\mathbf v_{\text{odo}}^{b}-v^n\Big)
\tag{X.12}
$$

残差依赖关系如下：

（1）依赖第$k$帧状态$q_k,v_k,b_{g,k}$；

（2）依赖全局标定参数$s,\alpha_p,\alpha_y,\mathbf l_{\text{OD}}$。

### X.3.6 残差对各量的雅可比

**（a）对安装角$\alpha_p,\alpha_y$**

由式(X.8)求偏导，得到

$$
\frac{\partial\mathbf c_x}{\partial\alpha_p}=
\begin{bmatrix}
-\cos\alpha_y\sin\alpha_p\\
\sin\alpha_y\sin\alpha_p\\
-\cos\alpha_p
\end{bmatrix},
\qquad
\frac{\partial\mathbf c_x}{\partial\alpha_y}=
\begin{bmatrix}
-\sin\alpha_y\cos\alpha_p\\
-\cos\alpha_y\cos\alpha_p\\
0
\end{bmatrix}
\tag{X.13}
$$

于是

$$
\frac{\partial\mathbf r_{\text{odo}}}{\partial\alpha_p}
=
\frac{1}{\sigma_{\text{odo}}}
R_b^n\,(1+s)\frac{dS}{dt}\,
\frac{\partial\mathbf c_x}{\partial\alpha_p}
\tag{X.14}
$$

$$
\frac{\partial\mathbf r_{\text{odo}}}{\partial\alpha_y}
=
\frac{1}{\sigma_{\text{odo}}}
R_b^n\,(1+s)\frac{dS}{dt}\,
\frac{\partial\mathbf c_x}{\partial\alpha_y}
\tag{X.15}
$$

**（b）对刻度因子$s$**

$$
\frac{\partial\mathbf r_{\text{odo}}}{\partial s}
=
\frac{1}{\sigma_{\text{odo}}}
R_b^n\,\frac{dS}{dt}\,\mathbf c_x(\boldsymbol\alpha)
\tag{X.16}
$$

**（c）对杆臂$\mathbf l_{\text{OD}}$**

$$
\frac{\partial\mathbf r_{\text{odo}}}{\partial\mathbf l_{\text{OD}}}
=
\frac{1}{\sigma_{\text{odo}}}
R_b^n\,[\boldsymbol\omega_{ib}^{b}]_\times
\tag{X.17}
$$

其中$[\cdot]_\times$为反对称矩阵算子。

**（d）对姿态$q_k$**

设$q_k$右扰动为$q_k\otimes\text{Exp}(\delta\phi)$，则

$$
\frac{\partial R_b^n}{\partial\delta\phi}
=
-R_b^n\,[\mathbf v_{\text{odo}}^{b}]_\times
\tag{X.18}
$$

于是

$$
\frac{\partial\mathbf r_{\text{odo}}}{\partial\delta\phi}
=
-\frac{1}{\sigma_{\text{odo}}}
R_b^n\,[\mathbf v_{\text{odo}}^{b}]_\times
\tag{X.19}
$$

**（e）对速度$v_k$**

$$
\frac{\partial\mathbf r_{\text{odo}}}{\partial v_k}
=
-\frac{1}{\sigma_{\text{odo}}}\,I_3
\tag{X.20}
$$

**（f）对陀螺零偏$b_{g,k}$**

由于$\boldsymbol\omega_{ib}^{b}=d\theta_m/dt-b_{g,k}$，代入得

$$
\frac{\partial\mathbf r_{\text{odo}}}{\partial b_{g,k}}
=
-\frac{1}{\sigma_{\text{odo}}}
R_b^n\,[\mathbf l_{\text{OD}}]_\times
\tag{X.21}
$$

## X.4 双天线航向因子

### X.4.1 观测模型

双天线GNSS给出的航向$\psi_{\text{gnss}}$是天线基线在ENU下的方位角。由于基线相对IMU机体系存在安装偏置$\psi_{\text{off}}$，观测方程为

$$
\psi_{\text{gnss}}=\text{yaw}(q)+\psi_{\text{off}}+n_\psi,
\qquad n_\psi\sim\mathcal N(0,\sigma_\psi^2)
\tag{X.22}
$$

### X.4.2 航向提取

姿态四元数为$q=[q_0,q_1,q_2,q_3]=[w,x,y,z]$，对应旋转矩阵$R$。与姿态解算约定一致，航向定义为

$$
\text{yaw}(q)=\operatorname{atan2}(-R_{01},\ R_{11})
\tag{X.23}
$$

用四元数表示：

$$
\text{yaw}(q)=\operatorname{atan2}\big(2(q_0q_3-q_1q_2),\ q_0^2-q_1^2+q_2^2-q_3^2\big)
\tag{X.24}
$$

### X.4.3 残差定义

定义标准化残差

$$
r_\psi=
\frac{1}{\sigma_\psi}
\operatorname{wrap}\big(\text{yaw}(q)+\psi_{\text{off}}-\psi_{\text{gnss}}\big)
\tag{X.25}
$$

其中角度规整算子为

$$
\operatorname{wrap}(d)=\operatorname{atan2}(\sin d,\ \cos d)
\tag{X.26}
$$

**说明**：式(X.26)使用$\operatorname{atan2}(\sin,\cos)$而非条件判断语句，是因为它对自动微分连续可导，可避免在角度跳变点处导数不连续。

### X.4.4 残差对航向偏置的雅可比

由式(X.25)直接求导：

$$
\frac{\partial r_\psi}{\partial\psi_{\text{off}}}=\frac{1}{\sigma_\psi}
\tag{X.27}
$$

由式(X.27)可知，只要存在至少一帧有效航向观测，$\psi_{\text{off}}$就有非零梯度，即可被优化。

### X.4.5 残差对姿态的雅可比

设$q$右扰动为$q\otimes\text{Exp}(\delta\phi)$，则

$$
R'=R\big(I+[\delta\phi]_\times\big)
\tag{X.28}
$$

记$N=-R_{01}$，$D=R_{11}$，$\text{yaw}=\operatorname{atan2}(N,D)$，由链式法则：

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=
\frac{D\frac{\partial N}{\partial\delta\phi}-N\frac{\partial D}{\partial\delta\phi}}
{D^2+N^2}
\tag{X.29}
$$

在右扰动下

$$
\frac{\partial R(i,1)}{\partial\delta\phi}=\big(R(i,2),\ 0,\ -R(i,0)\big)
\tag{X.30}
$$

因此

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=
\frac{1}{R_{01}^2+R_{11}^2}
\big(-R_{11}R_{02}+R_{01}R_{12},\ 0,\ R_{11}R_{00}-R_{01}R_{10}\big)
\tag{X.31}
$$

将$R$的欧拉角展开式代入，可化简为

$$
\frac{\partial\text{yaw}}{\partial\delta\phi}
=
\left(
-\frac{\tan(\text{roll})}{\cos(\text{pitch})},\
0,\
\frac{\cos(\text{roll})}{\cos(\text{pitch})}
\right)
\tag{X.32}
$$

于是

$$
\frac{\partial r_\psi}{\partial\delta\phi}
=
\frac{1}{\sigma_\psi}\cdot
\frac{\partial\text{yaw}}{\partial\delta\phi}
\tag{X.33}
$$

## X.5 全局法方程累积与迭代更新

### X.5.1 法方程累积

设滑窗内共有$N$帧。所有残差块累积到全局法方程：

$$
H=\sum_{k=1}^{N}J_k^TJ_k,\qquad
b=\sum_{k=1}^{N}J_k^T\mathbf r_k
\tag{X.34}
$$

按状态部分与标定部分分块：

$$
H=
\begin{bmatrix}
H_{xx} & H_{xc}\\
H_{xc}^T & H_{cc}
\end{bmatrix},\qquad
b=
\begin{bmatrix}
b_x\\ b_c
\end{bmatrix}
\tag{X.35}
$$

### X.5.2 标定部分的信息矩阵

由里程计与航向因子的雅可比累积得到

$$
H_{cc}=
\sum_{k\in\mathcal O}
\left(\frac{\partial\mathbf r_{\text{odo},k}}{\partial c}\right)^T
\left(\frac{\partial\mathbf r_{\text{odo},k}}{\partial c}\right)
+
\sum_{k\in\mathcal Y}
\left(\frac{\partial r_{\psi,k}}{\partial c}\right)^T
\left(\frac{\partial r_{\psi,k}}{\partial c}\right)
\tag{X.36}
$$

其中$\mathcal O$为有里程计观测的帧集合，$\mathcal Y$为有航向观测的帧集合。

由式(X.36)可见，$H_{cc}$是所有帧的雅可比平方和。即使单帧的雅可比很小，多帧累积后$H_{cc}$也能充分大，标定参数就可观。这是将标定量作为全局参数块共享的数学依据。

### X.5.3 Schur补与边缘化处理

当滑窗超限时，最旧帧的状态$x_0$通过Schur补从法方程中消去。对状态部分做Schur补，得到仅关于标定参数的等价法方程：

$$
\big(H_{cc}-H_{xc}^TH_{xx}^{-1}H_{xc}\big)\,\delta c
=
b_c-H_{xc}^TH_{xx}^{-1}b_x
\tag{X.37}
$$

记

$$
H_{cc}^*=H_{cc}-H_{xc}^TH_{xx}^{-1}H_{xc}
\tag{X.38}
$$

$$
b_c^*=b_c-H_{xc}^TH_{xx}^{-1}b_x
\tag{X.39}
$$

则标定参数的最优更新为

$$
\delta c=(H_{cc}^*+\lambda I)^{-1}\,b_c^*
\tag{X.40}
$$

其中$\lambda$为Levenberg-Marquardt阻尼因子。

**关键特征**：标定参数块$c$不参与边缘化，始终保留在优化变量集合中。这是由标定量的物理属性决定的：标定参数在整个会话中保持不变，不属于任何单一帧的状态，因此不随帧滑出而消去。

### X.5.4 Levenberg-Marquardt迭代更新

第$t$次迭代解线性方程

$$
(H+\lambda I)\begin{bmatrix}\delta x\\ \delta c\end{bmatrix}
=
-\begin{bmatrix}b_x\\ b_c\end{bmatrix}
\tag{X.41}
$$

状态更新用流形加法：

$$
x^{(t+1)}=x^{(t)}\boxplus\delta x
\tag{X.42}
$$

标定参数更新用加性：

$$
c^{(t+1)}=c^{(t)}+\delta c
\tag{X.43}
$$

## X.6 标定模式与导航模式的统一

### X.6.1 标定模式

标定模式下标定参数块不被固定：

```cpp
problem.AddParameterBlock(calib_data_.data(), NUM_CALIB);
if (!opt_.calib_mode) {
    problem.SetParameterBlockConstant(calib_data_.data());
}
```

## X.6 标定模式与导航模式的统一

### X.6.1 标定模式

标定模式下标定参数块不被固定。在程序实现中，先通过 `AddParameterBlock` 添加标定参数块，再通过 `calib_mode` 开关判断是否调用 `SetParameterBlockConstant`。当 `calib_mode` 为真时，不调用 `SetParameterBlockConstant`，标定参数块作为自由变量参与优化。此时标定参数与状态一起被优化：

$$
\min_{x,\, c} \; \sum_{k} \big\| \mathbf{r}_k(x, c) \big\|^2
\tag{X.47}
$$

### X.6.2 导航模式

导航模式下标定参数被固定为已知值 $c^{\star}$。当 `calib_mode` 为假时，程序调用 `SetParameterBlockConstant` 将标定参数块固定，作为常量参与残差计算：

$$
\min_{x} \; \sum_{k} \big\| \mathbf{r}_k(x, c^{\star}) \big\|^2
\tag{X.48}
$$

因子仍然引用标定块（作为已知常量），但不更新。

### X.6.3 数学上的统一性

两种模式共用同一套残差函数，唯一差别是参数块是否进入优化变量集合。如表X.3所示。

**表X.3 标定模式与导航模式的对比**

| 模式 | 残差函数 | 标定块 | 优化变量 |
|---|---|---|---|
| 标定 | $\mathbf{r}_k(x, c)$ | 自由 | $x,\ c$ |
| 导航 | $\mathbf{r}_k(x, c^{\star})$ | 固定 | $x$ |

这一设计避免了维护两套代码分支，同一套因子函数在两种模式下都工作，仅通过参数块是否固定来区分，便于工程实现与维护。
