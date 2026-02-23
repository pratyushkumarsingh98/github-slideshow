# -*- coding: utf-8 -*-
"""
Two-heater, dual adaptive PID (December logic), FIXED dt=0.01 s.
ZOH timing: commands computed at k are applied at k+1 (NO RTD2->RTD3 within same step).

Fixes applied:
- EPS moved before first use (prevents NameError in nozzle functions).
- RAW PID traces are labeled as duty (not Watts) to match actuator logic.
- More explicit CoolProp phase handling for non-liquid/gas regions.
"""

import math

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# =============================================================================
# GLOBAL NUMERICAL SAFETY
# =============================================================================
EPS = 1e-30  # tiny number to prevent division-by-zero

# =========================
# PWM / Duty-cycle actuation (paper-like power spikes)
# =========================
PWM_ON = False
PWM_PERIOD = 0.50
P_MAX_PRE = None
P_MAX_MAIN = None


def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def duty_from_power(P_cmd, P_max):
    return _clip01(P_cmd / (P_max + EPS))


def pwm_gate(t, duty, period):
    phase = (t % period) / period
    return 1.0 if phase < duty else 0.0


# =============================================================================
# USER SWITCHES
# =============================================================================
FIXED_DT = 0.01
SUPERVISOR_ON = True
PRINT_EVERY = 20

# =============================================================================
# DISCRETIZATION / RUNTIME
# =============================================================================
n_it = 200
n_steps = 5000

# =============================================================================
# GEOMETRY (December)
# =============================================================================
H_int = 160e-6

L_inlet = 1200e-6
W_inlet = 500e-6

channelWidth = 120e-6
channelNum = 3
finWidth = 70e-6
finNum = 2
L_uCh = 6200e-6

L_intDiv = 2200e-6
W_intDiv1 = channelNum * channelWidth + finNum * finWidth
W_intDiv2 = 3000e-6
alpha_intDiv1 = math.degrees(math.atan(0.5 * (W_intDiv2 - W_intDiv1) / L_intDiv))

L_int = 17160e-6
dx = L_int / n_it

L_ext = 19900e-6
W_ext = 5000e-6
H_ext = 500e-6
A_ext = L_ext * W_ext * 2 + H_ext * L_ext * 2 + H_ext * W_ext * 2

# =============================================================================
# CONSTANTS / OPERATING POINT
# =============================================================================
specie = "Water"
T_ambient = 298.0
h_conv_ext = 100.0
sigma_stef_boltz = 5.670373e-8
k_Si = 149.0
em = 0.05

T_wall_start = 150 + 273.15
T_fl_start = 298.0

p_start = 370000.0
mass_flow = 1.90e-6

# =============================================================================
# NOZZLE MODEL
# =============================================================================
p_ambient = 101325.0
g0 = 9.81

W_t = 70e-6
W_e = 300e-6
A_t = W_t * H_int
A_e = W_e * H_int
throatCurv = 35e-6
div_length = 1200e-6


def quality_from_hP(h_val, p_val, specie="Water", clip=True):
    p_use = float(max(p_val, 1e3))
    hL = CP.PropsSI("H", "P", p_use, "Q", 0, specie)
    hV = CP.PropsSI("H", "P", p_use, "Q", 1, specie)
    denom = hV - hL
    if abs(denom) < 1e-12:
        return 0.0
    x = (h_val - hL) / denom
    if clip:
        x = np.clip(x, 0.0, 1.0)
    return float(x)


def nozzle_metrics(T0, P0, x_vap, specie="Water"):
    if P0 <= p_ambient * 1.001:
        return 0.0, 0.0, 0.0

    T_sat = CP.PropsSI("T", "P", P0, "Q", 0, specie)

    cp_L = CP.PropsSI("Cpmass", "T", T_sat, "Q", 0, specie)
    cv_L = CP.PropsSI("Cvmass", "T", T_sat, "Q", 0, specie)
    cp_V = CP.PropsSI("Cpmass", "T", T_sat, "Q", 1, specie)
    cv_V = CP.PropsSI("Cvmass", "T", T_sat, "Q", 1, specie)

    cp = (1.0 - x_vap) * cp_L + x_vap * cp_V
    cv = (1.0 - x_vap) * cv_L + x_vap * cv_V
    gamma = float(cp / (cv + EPS))
    Rs = float(cp - cv)

    T_star = T0 * 2.0 / (gamma + 1.0)
    P_star = P0 * (T_star / T0) ** (gamma / (gamma - 1.0))
    rho_star = P_star / (Rs * T_star + EPS)
    a_star = math.sqrt(gamma * Rs * T_star)
    m_dot_ideal = rho_star * a_star * A_t

    mu_L = CP.PropsSI("V", "T", T_sat, "Q", 0, specie)
    mu_V = CP.PropsSI("V", "T", T_sat, "Q", 1, specie)
    mu_mix = (1.0 - x_vap) * mu_L + x_vap * mu_V
    mu_star = mu_mix * (T_star / T_sat) ** 0.7

    Re_star = rho_star * a_star * W_t / (mu_star + 1e-8)

    rc = throatCurv
    f_gamma = 0.97 + 0.86 * gamma
    base1 = (rc + 0.05 * 0.5 * W_t) / (rc + 0.75 * 0.5 * W_t + EPS)
    base2 = (rc + 0.10 * 0.5 * W_t) / (0.5 * W_t + EPS)
    Cd_noz = base1 ** 0.019 * (1.0 - base2 ** 0.21 * Re_star ** -0.5 * f_gamma)
    Cd_noz = float(np.clip(Cd_noz, 0.0, 1.0))

    m_dot_act = Cd_noz * m_dot_ideal

    M_e_sq = ((P0 / p_ambient) ** ((gamma - 1.0) / gamma) - 1.0) * 2.0 / (gamma - 1.0)
    M_e = math.sqrt(max(M_e_sq, 0.0))
    T_e = T0 / (1.0 + 0.5 * (gamma - 1.0) * M_e**2)
    a_e = math.sqrt(gamma * Rs * T_e)
    u_e = M_e * a_e

    rho_e = p_ambient / (Rs * T_e + EPS)
    mu_e = mu_star * (T_e / T_star) ** 0.7
    Re_e = rho_e * u_e * div_length / (mu_e + 1e-8)

    delta_star_bar = 0.048 * div_length / (Re_e**0.2 + EPS)
    theta_bar = 0.037 * div_length / (Re_e**0.2 + EPS)
    S_bar = 0.048 / 0.037
    S_delta = S_bar * (1.0 + 0.113 * M_e**2) + 0.290 * M_e**2
    theta = theta_bar * (
        1.0 - (0.92 * M_e**2 / (7.09 + M_e**2 + EPS)) * math.tanh(1.49 * (S_bar - 0.9))
    )
    delta_star = S_delta * theta
    eta_u = 1.0 - (2.0 * delta_star / (W_e + EPS)) * (1.0 + theta / (delta_star + 1e-8))
    eta_u = float(np.clip(eta_u, 0.0, 1.0))

    return float(m_dot_act), float(u_e), float(eta_u)


q_max = 20.0
deadband = 1.0


# =============================================================================
# HELPERS
# =============================================================================
def find_nearest_index(array, value):
    return int(np.argmin(np.abs(array - value)))


def calculate_parameters(
    n_it,
    L_int,
    W_intDiv2,
    alpha_intDiv1,
    W_inlet,
    L_inlet,
    L_uCh,
    L_intDiv,
    channelNum,
    channelWidth,
    finNum,
    finWidth,
    H_int,
):
    dx_loc = L_int / n_it
    x = np.arange(0, n_it * dx_loc, dx_loc)

    dy_Div = dx_loc * np.tan(np.deg2rad(alpha_intDiv1))

    W_int = np.zeros(len(x))
    W_int[0] = W_inlet

    for ii in range(1, len(x)):
        if x[ii] < L_inlet:
            W_int[ii] = W_inlet
        elif (x[ii] >= L_inlet) and (x[ii] < (L_inlet + L_uCh)):
            W_int[ii] = channelNum * channelWidth
        elif (x[ii] >= (L_inlet + L_uCh)) and (x[ii] < (L_inlet + L_uCh + L_intDiv)):
            W_int[ii] = W_int[ii - 1] + 2 * dy_Div
        else:
            W_int[ii] = W_intDiv2

    for ii in range(len(x)):
        if (x[ii] >= (L_inlet + L_uCh)) and (x[ii] < (L_inlet + L_uCh + L_intDiv)):
            W_int[ii] += finNum * finWidth

    A_cs = np.zeros(len(x))
    Perim = np.zeros(len(x))
    Dh = np.zeros(len(x))

    for ii in range(len(x)):
        if L_inlet <= x[ii] < (L_inlet + L_uCh):
            A_cs[ii] = W_int[ii] * H_int
            Perim[ii] = 2 * W_int[ii] + channelNum * H_int * 2
            Dh[ii] = 4 * A_cs[ii] / (Perim[ii] + EPS)
        else:
            A_cs[ii] = W_int[ii] * H_int
            Perim[ii] = 2 * W_int[ii] + H_int * 2
            Dh[ii] = 4 * A_cs[ii] / (Perim[ii] + EPS)

    return x, W_int, A_cs, Perim, Dh


def calculation_P_cond_struct(k_mat, T_wall, T_amb, A_ext_local, L_ext_local):
    return k_mat * (T_wall - T_amb) * A_ext_local / (L_ext_local + EPS)


def calculation_P_rad(T_surf, T_amb, em, sigma_stef_boltz, A_ext_local):
    return (T_surf**4 - T_amb**4) * em * sigma_stef_boltz * A_ext_local


def calculation_P_conv(h_conv, T_wall, T_amb, A_ext_local):
    return h_conv * A_ext_local * (T_wall - T_amb)


# =============================================================================
# SETPOINTS — Option B (stepwise)
# =============================================================================
def setpoint_rtd2(t):
    if t < 10:
        return 60 + 273.15
    if t < 20:
        return 80 + 273.15
    if t < 30:
        return 100 + 273.15
    if t < 40:
        return 110 + 273.15
    return 120 + 273.15


def setpoint_rtd3(t):
    if t < 10:
        return 60 + 273.15
    if t < 20:
        return 80 + 273.15
    if t < 30:
        return 100 + 273.15
    if t < 40:
        return 120 + 273.15
    return 150 + 273.15


# =============================================================================
# ADAPTIVE PID
# =============================================================================
class AdaptivePIDNN:
    def __init__(
        self,
        kp0=0.0005,
        ki0=0.005,
        kd0=0.0001,
        adapt_rate_p=0.1,
        adapt_rate_i=0.1,
        adapt_rate_d=0.05,
        big_error_threshold=5.0,
        integral_clip=1e6,
    ):
        self.kp0 = kp0
        self.ki0 = ki0
        self.kd0 = kd0
        self.kp = kp0
        self.ki = ki0
        self.kd = kd0
        self.adapt_rate_p = adapt_rate_p
        self.adapt_rate_i = adapt_rate_i
        self.adapt_rate_d = adapt_rate_d
        self.big_error_threshold = big_error_threshold
        self.integral = 0.0
        self.last_error = 0.0
        self.integral_clip = float(integral_clip)

    def update(self, current_value, setpoint, dt):
        error = setpoint - current_value

        # Leaky integrator reduces long-memory lock-in after large overshoot events.
        self.integral *= 0.999
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -self.integral_clip, self.integral_clip))

        derivative = (error - self.last_error) / dt if dt > 0.0 else 0.0
        u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Sign-consistent anti-windup: if control action fights the current error,
        # bleed integral in the opposite direction and recompute u.
        if error > 0.0 and u < 0.0:
            self.integral = max(self.integral, 0.0)
            u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        elif error < 0.0 and u > 0.0:
            self.integral = min(self.integral, 0.0)
            u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        abs_e = abs(error)
        abs_last_e = abs(self.last_error)
        big_err = self.big_error_threshold

        if abs_e > big_err and abs_e >= abs_last_e:
            self.kp *= 1.0 + self.adapt_rate_p * dt
            self.ki *= 1.0 + self.adapt_rate_i * dt

        if np.sign(error) != np.sign(self.last_error) and abs_last_e > big_err:
            self.kp *= 1.0 - self.adapt_rate_p * dt
            self.ki *= 1.0 - self.adapt_rate_i * dt
            self.kd *= 1.0 + self.adapt_rate_d * dt

        self.kp = float(np.clip(self.kp, 0.2 * self.kp0, 5.0 * self.kp0))
        self.ki = float(np.clip(self.ki, 0.2 * self.ki0, 5.0 * self.ki0))
        self.kd = float(np.clip(self.kd, 0.2 * self.kd0, 5.0 * self.kd0))

        self.last_error = error
        return float(u)


# =============================================================================
# SUPERVISOR
# =============================================================================
def supervisor(Ppre_cmd, Pmain_cmd, e2, e3, deadband_K=1.0, scale_min=0.3):
    if not SUPERVISOR_ON:
        return Ppre_cmd, Pmain_cmd
    if abs(e2) < deadband_K and abs(e3) < deadband_K:
        return max(Ppre_cmd * scale_min, 0.0), max(Pmain_cmd * scale_min, 0.0)
    return Ppre_cmd, Pmain_cmd


def allocate_min_power(Ppre_cmd, Pmain_cmd, e2, e3, losses_pre, losses_main, deadband_K=1.0):
    """Energy-aware allocator.

    - Prioritize main-heater power for RTD3 tracking.
    - Disable preheater unless both sections are cold and RTD3 still needs heat.
    - In the near-setpoint region, only apply loss-compensation power.
    - Enforce minimum RTD3 recovery power when RTD3 is far below setpoint.
    """
    # Any RTD3 overshoot: stop active heating to avoid oscillatory re-heating.
    if e3 < -deadband_K:
        return 0.0, 0.0

    Ppre = max(Ppre_cmd, 0.0)
    Pmain = max(Pmain_cmd, 0.0)

    # Near setpoint: only hold estimated thermal losses.
    if abs(e3) <= 2.0 * deadband_K and abs(e2) <= 2.0 * deadband_K:
        return float(np.clip(losses_pre, 0.0, q_max)), float(np.clip(losses_main, 0.0, q_max))

    # Guarantee a non-zero recovery action for RTD3 when far below target,
    # even if PID raw output briefly goes negative due to integral history.
    if e3 > 3.0 * deadband_K:
        min_recovery = max(0.10 * q_max, losses_main)
        Pmain = max(Pmain, min_recovery)

    # RTD3 gets primary authority; preheater acts only as assist when both are cold.
    if not (e3 > 2.0 * deadband_K and e2 > deadband_K):
        Ppre = 0.0

    return float(np.clip(Ppre, 0.0, q_max)), float(np.clip(Pmain, 0.0, q_max))


# =============================================================================
# TRANSIENT PLANT
# =============================================================================
def plant_step_transient(
    h,
    p,
    x,
    W_int,
    A_cs,
    Perim,
    Dh,
    p0,
    T_inlet,
    specie,
    L_inlet,
    L_uCh,
    L_int,
    dx,
    mass_flow,
    P_pre,
    P_main,
    dt_fixed,
):
    n = len(h)
    rtd2 = find_nearest_index(x, L_inlet + L_uCh)
    rtd3 = find_nearest_index(x, L_int)

    p[0] = float(max(p0, 1e3))
    h[0] = float(CP.PropsSI("H", "P", p[0], "T", T_inlet, specie))

    n_pre = max(rtd2, 1)
    n_main = max(rtd3 - rtd2, 1)

    Q = np.zeros(n)
    for i in range(n):
        if i < rtd2:
            Q[i] = P_pre / n_pre
        elif i < rtd3:
            Q[i] = P_main / n_main
        else:
            Q[i] = 0.0

    T_fl = np.zeros(n)
    T_w = np.zeros(n)

    h_new = h.copy()
    p_new = p.copy()

    for i in range(1, n):
        p_i = float(max(p[i], 1e3))

        try:
            T_i = CP.PropsSI("T", "P", p_i, "H", h[i], specie)
        except Exception:
            hL = CP.PropsSI("H", "P", p_i, "Q", 0, specie)
            hV = CP.PropsSI("H", "P", p_i, "Q", 1, specie)
            h[i] = float(np.clip(h[i], hL - 2e5, hV + 2e5))
            T_i = CP.PropsSI("T", "P", p_i, "H", h[i], specie)

        rho = CP.PropsSI("D", "P", p_i, "H", h[i], specie)

        phase = CP.PhaseSI("P", p_i, "HMASS", h[i], specie)
        phase_l = phase.lower()
        if "liquid" in phase_l and "twophase" not in phase_l:
            x_v = 0.0
        elif any(tag in phase_l for tag in ["gas", "supercritical", "supercritical_gas"]):
            x_v = 1.0
        else:
            rhoV = CP.PropsSI("D", "P", p_i, "Q", 1, specie)
            rhoL = CP.PropsSI("D", "P", p_i, "Q", 0, specie)
            denom = rhoL - rhoV
            x_v = float(np.clip((rhoL - rho) / denom, 0.0, 1.0)) if abs(denom) > 0 else 0.0

        kL = CP.PropsSI("L", "P", p_i, "Q", 0, specie)
        kV = CP.PropsSI("L", "P", p_i, "Q", 1, specie)
        k_mix = kV * x_v + kL * (1.0 - x_v)

        muL = CP.PropsSI("V", "P", p_i, "Q", 0, specie)
        muV = CP.PropsSI("V", "P", p_i, "Q", 1, specie)
        mu_mix = muV * x_v + muL * (1.0 - x_v)

        A_flow = float(W_int[i] * H_int)
        v = mass_flow / (rho * A_flow + EPS)
        CFL = float(np.clip(v * dt_fixed / (dx + EPS), 0.0, 1.0))

        dh_src = CFL * (Q[i] / (mass_flow + EPS))
        h_new[i] = h[i] - CFL * (h[i] - h[i - 1]) + dh_src

        dp = 12.0 * mu_mix * (v / (H_int**2 + EPS)) * dx
        p_new[i] = float(max(p_new[i - 1] - dp, 1e3))

        Nu = 4.96
        hc = Nu * k_mix / (Dh[i] + EPS)

        if i < rtd2:
            qflux = Q[i] / (Perim[i] * dx + EPS)
        elif i < rtd3:
            qflux = Q[i] / (W_int[i] * dx + EPS)
        else:
            qflux = 0.0

        try:
            T_new = CP.PropsSI("T", "P", p_new[i], "H", h_new[i], specie)
        except Exception:
            p_tmp = float(max(p_new[i], 1e3))
            hL2 = CP.PropsSI("H", "P", p_tmp, "Q", 0, specie)
            hV2 = CP.PropsSI("H", "P", p_tmp, "Q", 1, specie)
            h_new[i] = float(np.clip(h_new[i], hL2 - 2e5, hV2 + 2e5))
            T_new = CP.PropsSI("T", "P", p_tmp, "H", h_new[i], specie)

        T_fl[i] = float(T_new)
        T_w[i] = qflux / (hc + EPS) + float(T_new)

    T_fl[0] = T_inlet
    T_w[0] = T_inlet

    return h_new, p_new, T_fl, T_w


# =============================================================================
# MAIN
# =============================================================================
def main():
    xg, Wg, Acs, Perim, Dh = calculate_parameters(
        n_it,
        L_int,
        W_intDiv2,
        alpha_intDiv1,
        W_inlet,
        L_inlet,
        L_uCh,
        L_intDiv,
        channelNum,
        channelWidth,
        finNum,
        finWidth,
        H_int,
    )
    rtd2 = find_nearest_index(xg, L_inlet + L_uCh)
    rtd3 = find_nearest_index(xg, L_int)

    frac_pre = float(np.clip(xg[rtd2] / L_ext, 0.0, 1.0)) if L_ext > 0 else 0.5
    frac_main = float(np.clip((xg[rtd3] - xg[rtd2]) / L_ext, 0.0, 1.0)) if L_ext > 0 else 0.5
    A_pre = A_ext * frac_pre
    A_main = A_ext * frac_main

    pid2 = AdaptivePIDNN()
    pid3 = AdaptivePIDNN()

    Ppre_applied = 0.0
    Pmain_applied = 0.0

    h0 = float(CP.PropsSI("H", "P", p_start, "T", T_fl_start, specie))
    h = np.ones(n_it) * h0
    p = np.ones(n_it) * float(p_start)

    time = np.zeros(n_steps)
    T2_hist = np.zeros(n_steps)
    Tw2_hist = np.zeros(n_steps)
    T3_hist = np.zeros(n_steps)
    Tw3_hist = np.zeros(n_steps)
    SP2_hist = np.zeros(n_steps)
    SP3_hist = np.zeros(n_steps)

    u2_raw = np.zeros(n_steps)
    u3_raw = np.zeros(n_steps)

    # adaptive PID gain histories
    kp2_hist = np.zeros(n_steps); ki2_hist = np.zeros(n_steps); kd2_hist = np.zeros(n_steps)
    kp3_hist = np.zeros(n_steps); ki3_hist = np.zeros(n_steps); kd3_hist = np.zeros(n_steps)

    Ppre_applied_hist = np.zeros(n_steps)
    Pmain_applied_hist = np.zeros(n_steps)
    d_pre_hist = np.zeros(n_steps)
    d_main_hist = np.zeros(n_steps)
    Ppre_cmd_next_hist = np.zeros(n_steps)
    Pmain_cmd_next_hist = np.zeros(n_steps)

    noz_T0 = np.zeros(n_steps)
    noz_P0 = np.zeros(n_steps)
    noz_xvap_raw = np.zeros(n_steps)
    noz_xvap_clip = np.zeros(n_steps)
    noz_mdot_act = np.zeros(n_steps)
    noz_ue = np.zeros(n_steps)
    noz_eta_u = np.zeros(n_steps)

    for k in tqdm(range(1, n_steps), desc="Simulating", dynamic_ncols=True):
        t = k * FIXED_DT
        time[k] = t

        h, p, T_fl, T_wall = plant_step_transient(
            h,
            p,
            xg,
            Wg,
            Acs,
            Perim,
            Dh,
            p_start,
            T_fl_start,
            specie,
            L_inlet,
            L_uCh,
            L_int,
            dx,
            mass_flow,
            Ppre_applied,
            Pmain_applied,
            FIXED_DT,
        )

        T2 = float(T_fl[rtd2])
        Tw2 = float(T_wall[rtd2])
        T3 = float(T_fl[rtd3])
        Tw3 = float(T_wall[rtd3])

        T0 = float(T3)
        P0 = float(p[rtd3])
        x_exit_raw = quality_from_hP(float(h[rtd3]), float(p[rtd3]), specie=specie, clip=False)
        x_exit_clip = quality_from_hP(float(h[rtd3]), float(p[rtd3]), specie=specie, clip=True)

        m_dot_act, u_e, eta_u = nozzle_metrics(T0, P0, x_exit_clip, specie=specie)

        noz_T0[k] = T0
        noz_P0[k] = P0
        noz_xvap_raw[k] = x_exit_raw
        noz_xvap_clip[k] = x_exit_clip
        noz_mdot_act[k] = m_dot_act
        noz_ue[k] = u_e
        noz_eta_u[k] = eta_u

        T2_hist[k] = T2
        Tw2_hist[k] = Tw2
        T3_hist[k] = T3
        Tw3_hist[k] = Tw3

        SP2 = float(setpoint_rtd2(t))
        SP3 = float(setpoint_rtd3(t))
        SP2_hist[k] = SP2
        SP3_hist[k] = SP3

        e2 = SP2 - T2
        e3 = SP3 - T3

        u2 = float(pid2.update(T2, SP2, FIXED_DT))
        u3 = float(pid3.update(T3, SP3, FIXED_DT))
        u2_raw[k] = u2
        u3_raw[k] = u3

        kp2_hist[k], ki2_hist[k], kd2_hist[k] = pid2.kp, pid2.ki, pid2.kd
        kp3_hist[k], ki3_hist[k], kd3_hist[k] = pid3.kp, pid3.ki, pid3.kd

        Tw_pre = float(np.mean(T_wall[: max(rtd2, 1)]))
        Tw_main = float(np.mean(T_wall[rtd2 : max(rtd3, rtd2 + 1)]))

        losses_pre = (
            calculation_P_rad(Tw_pre, T_ambient, em, sigma_stef_boltz, A_pre)
            + calculation_P_conv(h_conv_ext, Tw_pre, T_ambient, A_pre)
            + 0.001 * calculation_P_cond_struct(k_Si, Tw_pre, T_ambient, A_pre, L_ext)
        )
        losses_main = (
            calculation_P_rad(Tw_main, T_ambient, em, sigma_stef_boltz, A_main)
            + calculation_P_conv(h_conv_ext, Tw_main, T_ambient, A_main)
            + 0.001 * calculation_P_cond_struct(k_Si, Tw_main, T_ambient, A_main, L_ext)
        )
        losses_pre = float(max(losses_pre, 0.0))
        losses_main = float(max(losses_main, 0.0))

        if e2 > deadband:
            d_pre_cmd_next = float(np.clip(u2, 0.0, 1.0))
        elif e2 < -deadband:
            d_pre_cmd_next = 0.0
        else:
            d_pre_cmd_next = float(np.clip(losses_pre / (q_max + EPS), 0.0, 1.0))

        if e3 > deadband:
            d_main_cmd_next = float(np.clip(u3, 0.0, 1.0))
        elif e3 < -deadband:
            d_main_cmd_next = 0.0
        else:
            d_main_cmd_next = float(np.clip(losses_main / (q_max + EPS), 0.0, 1.0))

        Ppre_cmd_next = d_pre_cmd_next * float(q_max)
        Pmain_cmd_next = d_main_cmd_next * float(q_max)

        Ppre_cmd_next, Pmain_cmd_next = supervisor(Ppre_cmd_next, Pmain_cmd_next, e2, e3)
        Ppre_cmd_next, Pmain_cmd_next = allocate_min_power(
            Ppre_cmd_next,
            Pmain_cmd_next,
            e2,
            e3,
            losses_pre,
            losses_main,
            deadband_K=deadband,
        )

        Ppre_applied_hist[k] = Ppre_applied
        Pmain_applied_hist[k] = Pmain_applied
        Ppre_cmd_next_hist[k] = Ppre_cmd_next
        Pmain_cmd_next_hist[k] = Pmain_cmd_next

        if PRINT_EVERY and (k % PRINT_EVERY == 0 or k == n_steps - 1):
            print(
                f"k={k:5d} t={t:8.3f}s dt={FIXED_DT:0.4f}s | "
                f"SP2={SP2:7.2f}K T2_fl={T2:7.2f}K T2_w={Tw2:7.2f}K e2={e2:8.2f} | "
                f"SP3={SP3:7.2f}K T3_fl={T3:7.2f}K T3_w={Tw3:7.2f}K e3={e3:8.2f} | "
                f"u2_raw(duty)={u2:7.3f} u3_raw(duty)={u3:7.3f} | "
                f"Ppre_cmd(next)={Ppre_cmd_next:7.3f}W Pmain_cmd(next)={Pmain_cmd_next:7.3f}W | "
                f"Ppre_applied(now)={Ppre_applied:7.3f}W Pmain_applied(now)={Pmain_applied:7.3f}W"
            )

        _Pmax_pre = float(q_max) if P_MAX_PRE is None else float(P_MAX_PRE)
        _Pmax_main = float(q_max) if P_MAX_MAIN is None else float(P_MAX_MAIN)

        d_pre = duty_from_power(Ppre_cmd_next, _Pmax_pre)
        d_main = duty_from_power(Pmain_cmd_next, _Pmax_main)

        if PWM_ON:
            g_pre = pwm_gate(t, d_pre, PWM_PERIOD)
            g_main = pwm_gate(t, d_main, PWM_PERIOD)
            Ppre_applied = g_pre * _Pmax_pre
            Pmain_applied = g_main * _Pmax_main
        else:
            Ppre_applied = Ppre_cmd_next
            Pmain_applied = Pmain_cmd_next

        d_pre_hist[k] = d_pre
        d_main_hist[k] = d_main

    def plot_heater_panel(rtd_tag, T_fl_hist, T_w_hist, SP_hist, P_hist, kp_hist, ki_hist, kd_hist):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        ax1.plot(time, T_fl_hist, label=f"Fluid Temperature @ {rtd_tag} [K]")
        ax1.plot(time, T_w_hist, "--", label=f"Wall Temperature @ {rtd_tag} [K]")
        ax1.plot(time, SP_hist, "-.", label="Setpoint Temperature [K]")
        ax1.set_title(f"Temperature Profile Over Time ({rtd_tag})")
        ax1.set_ylabel("T [K]")
        ax1.grid(True)
        ax1.legend(loc="upper right")

        ax2.plot(time, P_hist, label="Heat Input (W)")
        e_total = float(np.trapz(P_hist, time))
        avg_power = float(np.mean(P_hist))
        ax2.text(
            0.60,
            0.62,
            f"Total Heat Input = {e_total:.2f} J\nAverage Power = {avg_power:.2f} W",
            transform=ax2.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        ax2.set_title(f"Heat Input Over Time ({rtd_tag})")
        ax2.set_ylabel("H_input [W]")
        ax2.grid(True)
        ax2.legend(loc="upper right")

        ax3.plot(time, kp_hist, label="Kp")
        ax3.plot(time, ki_hist, label="Ki")
        ax3.plot(time, kd_hist, label="Kd")
        ax3.set_title(f"PID Parameters Over Time ({rtd_tag})")
        ax3.set_xlabel("t [s]")
        ax3.set_ylabel("PID Values")
        ax3.grid(True)
        ax3.legend(loc="upper right")

        fig.tight_layout()

    plot_heater_panel("RTD2", T2_hist, Tw2_hist, SP2_hist, Ppre_applied_hist, kp2_hist, ki2_hist, kd2_hist)
    plot_heater_panel("RTD3", T3_hist, Tw3_hist, SP3_hist, Pmain_applied_hist, kp3_hist, ki3_hist, kd3_hist)

    plt.figure(figsize=(14, 10))

    cx1 = plt.subplot(3, 2, 1)
    cx1.plot(time, noz_T0, label="T0 [K]")
    cx1.set_title("Nozzle Inlet Temperature T0")
    cx1.set_xlabel("Time [s]")
    cx1.set_ylabel("T0 [K]")
    cx1.grid(True)

    cx2 = plt.subplot(3, 2, 2)
    cx2.plot(time, noz_P0 / 1e5, label="P0 [bar]")
    cx2.set_title("Nozzle Inlet Pressure P0")
    cx2.set_xlabel("Time [s]")
    cx2.set_ylabel("P0 [bar]")
    cx2.grid(True)

    cx3 = plt.subplot(3, 2, 3)
    cx3.plot(time, noz_xvap_raw, label="x_vap_exit RAW [-]")
    cx3.plot(time, noz_xvap_clip, "--", label="x_vap_exit clipped [-]")
    cx3.set_title("Vapour Quality at Heater Exit (post-heat-control)")
    cx3.set_xlabel("Time [s]")
    cx3.set_ylabel("x_vap_exit [-]")
    cx3.grid(True)
    cx3.legend()

    cx4 = plt.subplot(3, 2, 4)
    cx4.plot(time, noz_mdot_act * 1e9, label="m_dot_act [µg/s]")
    cx4.set_title("Effective Mass Flow Through Nozzle")
    cx4.set_xlabel("Time [s]")
    cx4.set_ylabel("m_dot_act [µg/s]")
    cx4.grid(True)

    cx5 = plt.subplot(3, 2, 5)
    cx5.plot(time, noz_ue, label="u_e [m/s]")
    cx5.set_title("Exit Velocity u_e")
    cx5.set_xlabel("Time [s]")
    cx5.set_ylabel("u_e [m/s]")
    cx5.grid(True)

    cx6 = plt.subplot(3, 2, 6)
    cx6.plot(time, noz_eta_u, label="eta_u [-]")
    cx6.set_title("Boundary-Layer Efficiency η_u")
    cx6.set_xlabel("Time [s]")
    cx6.set_ylabel("η_u [-]")
    cx6.set_ylim(-0.05, 1.05)
    cx6.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
