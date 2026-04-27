%% ========== 连续雨强回归数据集：快衰落干扰 + K0 慢漂 ==========
clc; clear;

%% --- 链路基础参数 ---
L_m    = 41.5;        % 雨区长度（米）
Pt_dBm = 20;          % 发射功率（dBm）
T_C    = 27;  % 温度（℃），若后续使用 RainExtincCoef* 可用
dt_s   = 0.05;        % 采样间隔（秒）

% 选择频点：仅支持 120 / 140 / 229 GHz
f_GHz = 229;
f_Hz  = f_GHz * 1e9;

% 快衰落设置
fD_Hz     = 0;        % 最大多普勒（Hz）；0 表示 i.i.d.（最简单，利于回归）
base_seed = 2026;     % 随机种子（整批可复现）
rng(base_seed);

%% --- 数据集设计 ---
num_samples = 20000;          % 总样本窗口数
win_s       = 20;             % 每个训练样本窗口长度（建议 10~30 s）
Nw          = round(win_s/dt_s); % 每个窗口的采样点数

% R(t) 的更新间隔（分段常值）
TR_s      = 20;                 % 若 TR_s == win_s，则每个窗口内 R 为常值
% 若 TR_s < win_s，则一个窗口内 R 会变化（任务更难）
NR_update = round(TR_s/dt_s); 

% K0 慢漂的相关时间常数
tauK0_s = 60;                 % K0 相关时间
rho     = exp(-dt_s/tauK0_s); % AR(1) 系数：rho = exp(-dt/tau)

%% --- K(R,f) 参数（把最终结果填在 get_K_params 里）---
[K0_lo, K0_hi, a_pos] = get_K_params(f_GHz);
% 其中 a_pos > 0，满足：KdB = K0 - a_pos * R

% AR(1) 的均值（取区间中心）
K0_mu = (K0_lo + K0_hi)/2;

% K0 慢漂强度（每步创新噪声，需调参）
% 粗略原则：希望在 ~tauK0 时间尺度内能覆盖区间的一部分，但不乱跳
K0_sigma_step = 0.03;   % 每个采样点的创新噪声（dB），典型 0.02~0.08

%% --- 存储（建议保存为 MAT，比 Excel 快很多）---
X    = zeros(num_samples, Nw, 'single'); % 特征：Prx_dBm 序列（每行一个窗口）
Y    = zeros(num_samples, 1,  'single'); % 标签：雨强 R（mm/h，连续值）
Meta = struct();                         % 可选：存一些元信息

% 初始化慢变量状态 K0
K0_state = K0_mu;  % 初始从区间中心开始

% %% ===== K0 慢漂诊断绘图：先跑一段看看像不像慢漂=====
% diag_enable = true;     % 开关：true=画图，false=不画
% diag_dur_s  = 300;      % 诊断时长（秒），建议 300~600
% diag_seed   = 20260;    % 诊断随机种子（不影响后面数据生成，可单独设）
% 
% if diag_enable
%     rng(diag_seed);
% 
%     Nd = round(diag_dur_s/dt_s);
%     t_diag = (0:Nd-1)' * dt_s;
% 
%     % 从区间中心开始（也可改为 K0_lo 或 K0_hi 看边界行为）
%     K0_state_diag = K0_mu;
%     K0_traj_diag  = zeros(Nd,1);
% 
%     for k = 1:Nd
%         epsk = K0_sigma_step * randn;
%         K0_state_diag = rho*K0_state_diag + (1-rho)*K0_mu + epsk;
% 
%         % 反射边界
%         if K0_state_diag < K0_lo
%             K0_state_diag = K0_lo + (K0_lo - K0_state_diag);
%         elseif K0_state_diag > K0_hi
%             K0_state_diag = K0_hi - (K0_state_diag - K0_hi);
%         end
%         % 安全夹紧
%         K0_state_diag = min(max(K0_state_diag, K0_lo), K0_hi);
% 
%         K0_traj_diag(k) = K0_state_diag;
%     end
% 
%     figure('Name','K0 slow drift diagnostic');
%     plot(t_diag, K0_traj_diag, 'LineWidth', 1.6); grid on; box on; hold on;
%     yline(K0_lo, '--', 'K0\_lo');
%     yline(K0_hi, '--', 'K0\_hi');
%     yline(K0_mu, ':',  'K0\_mu');
% 
%     xlabel('时间 (s)');
%     ylabel('K0 (dB)');
%     title(sprintf('K0慢漂诊断 | f=%.0f GHz | dt=%.3fs | \\tau=%.0fs | \\sigma_{step}=%.3f dB', ...
%         f_GHz, dt_s, tauK0_s, K0_sigma_step));
%     hold off;
% 
%     % ===== 可选：打印一些量化指标 =====
%     fprintf('\n[K0 DIAG] duration=%.1fs, mean=%.3f, std=%.3f, min=%.3f, max=%.3f\n', ...
%         diag_dur_s, mean(K0_traj_diag), std(K0_traj_diag), min(K0_traj_diag), max(K0_traj_diag));
% end
% 
% % 恢复主流程随机种子（保证后面数据集完全可复现）
% rng(base_seed);

%% --- 主生成循环 ---
for n = 1:num_samples

    % 1) 为当前窗口采样连续雨强 R（示例：均匀分布 U(1,50)）
    % R_label = 1 + (50-1)*rand;

    % 可选：混合分布，让低雨强更常见（更贴近现实）
    if rand < 0.7
        R_label = 1 + (10-1)*rand;   % 70% 样本落在 1~10
    else
        R_label = 1 + (50-1)*rand;   % 30% 样本落在 1~50
    end

    % 在窗口内生成 K0 的慢漂轨迹
    % K0在Nw个点上缓慢变化
    K0_traj = zeros(Nw,1);
    for k = 1:Nw
        epsk     = K0_sigma_step * randn;                 
        K0_state = rho*K0_state + (1-rho)*K0_mu + epsk;   % 轨迹更新
        % 保持在区间内
        if K0_state < K0_lo
            K0_state = K0_lo + (K0_lo - K0_state);
        elseif K0_state > K0_hi
            K0_state = K0_hi - (K0_state - K0_hi);
        end
        % 防止极端跳变
        K0_state = min(max(K0_state, K0_lo), K0_hi);
        K0_traj(k) = K0_state;
    end

    % 3) 计算窗口内 KdB(t)
    KdB_traj  = K0_traj - a_pos * R_label;     % (Nw x 1)
    Klin_traj = 10.^(KdB_traj/10);             % 线性域 K

    % 4) 由雨衰得到大尺度平均功率（窗口内 R 视为常值）
    % 雨滴谱影响雨衰
    A_MP_dB = RainExtincCoef1(R_label, f_GHz, T_C) * (L_m/1000);  % M-P模型
    % A_JW_dB = RainExtincCoef2(R_label, f_GHz, T_C) * (L_m/1000);  % J-W模型
    % A_Joss_dB = RainExtincCoef3(R_label, f_GHz, T_C) * (L_m/1000);  % Joss模型
    % A_JT_dB = RainExtincCoef5(R_label, f_GHz, T_C) * (L_m/1000);  % J-T模型
    % A_JD_dB = RainExtincCoef4(R_label, f_GHz, T_C) * (L_m/1000);  % J-D总雨衰 dB
    % A_ITU_dB = rainpl(L_m, f_Hz, R_label);
    Pbar_W   = 10.^((Pt_dBm - A_MP_dB - 30)/10); % 平均接收功率 (W)

    % 5) 生成快衰落：K 随时间变化的 Rician 衰落序列
    h = rician_h_series_timevaryingK(Nw, dt_s, Klin_traj, fD_Hz);

    % 6) 合成瞬时接收功率
    P_W   = Pbar_W .* abs(h).^2;
    P_dBm = 10*log10(P_W) + 30;

    % 7) 存储一个训练样本窗口
    X(n,:) = single(P_dBm(:)).';
    Y(n)   = single(R_label);

    if mod(n, 2000)==0
        fprintf('[%d/%d] R=%.2f | K0区间=[%.2f, %.2f]\n', n, num_samples, R_label, K0_lo, K0_hi);
    end
end

% 记录元信息
Meta.f_GHz     = f_GHz;
Meta.dt_s      = dt_s;
Meta.win_s     = win_s;
Meta.tauK0_s   = tauK0_s;
Meta.K0_lo     = K0_lo;
Meta.K0_hi     = K0_hi;
Meta.a_pos     = a_pos;
Meta.base_seed = base_seed;

% 保存为 MAT
save(sprintf('dataset_MP_Rreg_%dGHz_%ds_%dsamples.mat', f_GHz, round(win_s), num_samples), ...
     'X','Y','Meta','-v7.3');
fprintf('MAT 数据集已保存。\n');

% %% R取值可视化验证
% % 所有 R_label 存到了 Y（num_samples×1）
% fprintf('R统计: min=%.2f, p05=%.2f, p50=%.2f, p95=%.2f, max=%.2f\n', ...
%     min(Y), prctile(Y,5), prctile(Y,50), prctile(Y,95), max(Y));
% 
% figure; histogram(Y, 50); grid on; box on;
% xlabel('R (mm/h)'); ylabel('Count'); title('训练样本 R 分布直方图');
% 
% % 关键区间计数
% n1_10  = sum(Y>=1  & Y<=10);
% n10_30 = sum(Y>10 & Y<=30);
% n30_50 = sum(Y>30 & Y<=50);
% fprintf('区间计数: [1,10]=%d | (10,30]=%d | (30,50]=%d | 总=%d\n', ...
%     n1_10, n10_30, n30_50, numel(Y));

%% ---------------- 本地函数 ----------------
function [K0_lo, K0_hi, a_pos] = get_K_params(fGHz)
% 最终确定的 K0 区间与斜率 a_pos（正数）
% 关系式：KdB = K0 - a_pos * R
    switch round(fGHz)
        case 120
            K0_lo = 51.90; K0_hi = 52.55; a_pos = 0.61881;
        case 140
            K0_lo = 48.68; K0_hi = 52.07; a_pos = 0.21497;
        case 229
            K0_lo = 48.44; K0_hi = 51.91; a_pos = 0.27652;
        otherwise
            error('本数据生成器仅支持 120/140/229 GHz。');
    end
end

function h = rician_h_series_timevaryingK(N, dt_s, K_lin_traj, fD_Hz)
% 生成“随时间变化 K”的 Rician 衰落序列（逐点 K）
% 目标：近似保持 E{|h|^2} ≈ 1
    if isempty(fD_Hz) || fD_Hz<=0
        % i.i.d. CN(0,1)
        g = (randn(N,1) + 1j*randn(N,1))/sqrt(2);
        g = g / sqrt(mean(abs(g).^2));
    else
        % 简化的 Jakes 类方法：多正弦叠加
        M = 32; n = (0:N-1).';
        alpha = 2*pi*rand(M,1); phi = 2*pi*rand(M,1);
        g = zeros(N,1);
        for m = 1:M
            g = g + exp(1j*(2*pi*fD_Hz*cos(alpha(m)).*(n*dt_s) + phi(m)));
        end
        g = g/sqrt(M);
        g = g / sqrt(mean(abs(g).^2));
    end

    phi0 = 2*pi*rand;  % LOS 初相位（窗口内常值）
    h = zeros(N,1);

    for k = 1:N
        Kk  = max(K_lin_traj(k), 1e-6);
        mu  = sqrt(Kk/(Kk+1));
        sig = sqrt(1/(2*(Kk+1)));
        h(k) = mu*exp(1j*phi0) + sqrt(2)*sig*g(k);
    end
end