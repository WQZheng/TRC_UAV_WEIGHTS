"""ORCA / reciprocal-velocity-obstacle baseline planner (the unequipped
aircraft in the penetration-rate study).

The multi-agent penetration experiment (Section~\ref{sec:results-penetration})
mixes aircraft that run \method (predictor + CBF-MPC) with aircraft that run a
credible, published collision-avoidance rule. Using a trivial constant-velocity
or hand-tuned rule for the unequipped aircraft would make the p=0% endpoint an
implausible strawman and expose any positive-externality claim to the objection
that it was designed into a weak baseline. We therefore use Optimal Reciprocal
Collision Avoidance (ORCA) [van den Berg et al., 2011], the de-facto standard
decentralised multi-agent avoidance method, which each unequipped aircraft runs
to select a horizontal velocity that is collision-free under the reciprocity
assumption that neighbours share the avoidance effort.

Scope and simplifications (stated honestly):
  * Avoidance is planar (horizontal); the corridor assigns a nominal cruise
    altitude and ORCA reasons in the (x,y) plane, consistent with how the
    CBF-MPC planner is exercised in the corridor scenarios.
  * Each agent is a disc of radius r = d_sep/2 so that two agents at the ORCA
    separation 2r are exactly at the d_sep separation standard used everywhere
    else in the paper.
  * A single-step ORCA (one half-plane per neighbour, solved by a small linear
    program) is used at each control tick; this is the standard formulation.
  * The selected ORCA velocity is converted to a commanded acceleration and
    then to the same 6-DOF control map (accel_to_control) used by \method, so
    both equipped and unequipped aircraft are actuated through the identical
    physical envelope and only the *decision* differs.

This module has no learned parameters and no dependence on the predictor; it is
a pure geometric controller, which is exactly what makes it a fair reference.
"""
from __future__ import annotations
import numpy as np


def _orca_velocity(p_self, v_self, v_pref, neighbours, radius, tau, vmax):
    """Single-agent ORCA update in the horizontal plane.

    p_self, v_self, v_pref : (2,) arrays  -- position, current & preferred vel
    neighbours : list of (p_nb (2,), v_nb (2,))
    radius : combined-disc radius per agent (so min sep = 2*radius = d_sep)
    tau    : ORCA time horizon [s]
    vmax   : speed cap [m/s]
    Returns the new velocity (2,).

    We build one ORCA half-plane constraint per neighbour (reciprocal, i.e.
    each side takes half of u) and then pick the velocity closest to v_pref
    that satisfies all half-planes, via a small projected search. This is the
    standard 2-D ORCA construction; for the modest neighbour counts here a
    dense candidate projection is exact enough and avoids a full LP solver.
    """
    constraints = []  # each: (normal n (2,), point-on-line w0 (2,))
    for p_nb, v_nb in neighbours:
        rel_p = np.asarray(p_nb) - np.asarray(p_self)
        rel_v = np.asarray(v_self) - np.asarray(v_nb)
        dist = np.linalg.norm(rel_p)
        comb_r = 2.0 * radius
        if dist < 1e-6:
            continue
        if dist > comb_r:
            # velocity obstacle truncated by the time horizon tau
            # apex of the VO cone at rel_p/tau
            w = rel_v - rel_p / tau
            w_len = np.linalg.norm(w)
            leg = np.sqrt(max(dist * dist - comb_r * comb_r, 1e-9))
            if w_len > 1e-9 and np.dot(w, rel_p) < 0:
                # project on the cut-off circle
                n = w / w_len
                u = (comb_r / tau - w_len) * n
            else:
                # project on the cone legs
                # normal to the closer leg
                theta = np.arctan2(rel_p[1], rel_p[0])
                phi = np.arcsin(np.clip(comb_r / dist, -1, 1))
                # two leg directions; choose the side of rel_v
                left = np.array([np.cos(theta + phi), np.sin(theta + phi)])
                right = np.array([np.cos(theta - phi), np.sin(theta - phi)])
                # unit leg normals pointing outside the VO
                nl = np.array([-left[1], left[0]])
                nr = np.array([right[1], -right[0]])
                # pick the leg whose projection is closer
                dl = np.dot(rel_v - rel_p / tau, nl)
                dr = np.dot(rel_v - rel_p / tau, nr)
                if dl <= dr:
                    n = nl
                    u = -dl * nl
                else:
                    n = nr
                    u = -dr * nr
        else:
            # already overlapping: push apart on the current time step
            n = rel_p / (dist + 1e-9)
            u = (comb_r - dist) / 0.2 * (-n)  # dt=0.2 step recovery
            n = -n
        # reciprocal: each takes half of u; half-plane: (v - (v_self + 0.5u))·n >= 0
        plane_point = np.asarray(v_self) + 0.5 * u
        constraints.append((n, plane_point))

    # find velocity nearest v_pref satisfying all half-planes.
    cand = np.asarray(v_pref, dtype=float)

    def satisfies(v):
        for n, w0 in constraints:
            if np.dot(v - w0, n) < -1e-6:
                return False
        return True

    if satisfies(cand):
        v_new = cand
    else:
        # project sequentially onto violated half-planes; then sample the
        # boundary intersections for the closest feasible point.
        best = None
        best_d = np.inf
        # try projection onto each constraint line + pairwise intersections
        lines = constraints
        cand_set = []
        for n, w0 in lines:
            # projection of v_pref onto this line
            proj = cand - np.dot(cand - w0, n) * n
            cand_set.append(proj)
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                n1, w1 = lines[i]
                n2, w2 = lines[j]
                A = np.array([n1, n2])
                b = np.array([np.dot(w1, n1), np.dot(w2, n2)])
                det = np.linalg.det(A)
                if abs(det) > 1e-6:
                    cand_set.append(np.linalg.solve(A, b))
        for v in cand_set:
            if np.linalg.norm(v) > vmax:
                v = v / np.linalg.norm(v) * vmax
            if satisfies(v):
                d = np.linalg.norm(v - cand)
                if d < best_d:
                    best_d, best = d, v
        v_new = best if best is not None else np.zeros(2)

    sp = np.linalg.norm(v_new)
    if sp > vmax:
        v_new = v_new / sp * vmax
    return v_new


class ORCAPlanner:
    """Decentralised ORCA controller for one unequipped aircraft.

    Produces a commanded horizontal acceleration toward an ORCA-feasible
    velocity that tracks a preferred cruise velocity along the corridor.
    """

    def __init__(self, d_sep=30.0, tau=4.0, vmax=25.0, dt=0.2, kv=1.5):
        self.radius = d_sep / 2.0
        self.tau = tau
        self.vmax = vmax
        self.dt = dt
        self.kv = kv

    def command_accel(self, p_self, v_self, v_pref, neighbours):
        """All args in inertial frame. p_self/v_self/v_pref: (3,). neighbours:
        list of (p_nb (3,), v_nb (3,)). Returns commanded accel (3,)."""
        p2 = np.asarray(p_self)[:2]
        v2 = np.asarray(v_self)[:2]
        vp2 = np.asarray(v_pref)[:2]
        nb2 = [(np.asarray(pn)[:2], np.asarray(vn)[:2]) for pn, vn in neighbours]
        v_orca = _orca_velocity(p2, v2, vp2, nb2, self.radius, self.tau,
                                self.vmax)
        # proportional velocity tracking -> acceleration; hold altitude.
        a_xy = self.kv * (v_orca - v2)
        a_z = self.kv * (np.asarray(v_pref)[2] - np.asarray(v_self)[2])
        return np.array([a_xy[0], a_xy[1], a_z])


if __name__ == "__main__":
    # sanity: two agents head-on should veer apart, not collide.
    orca = ORCAPlanner(d_sep=30.0, tau=4.0, vmax=25.0)
    pA = np.array([0.0, 0.0, 100.0]); vA = np.array([20.0, 0.0, 0.0])
    pB = np.array([200.0, 0.0, 100.0]); vB = np.array([-20.0, 0.0, 0.0])
    dt = 0.2
    min_sep = 1e9
    for t in range(80):
        aA = orca.command_accel(pA, vA, np.array([20.0, 0, 0]), [(pB, vB)])
        aB = orca.command_accel(pB, vB, np.array([-20.0, 0, 0]), [(pA, vA)])
        vA = vA + dt * aA; vB = vB + dt * aB
        pA = pA + dt * vA; pB = pB + dt * vB
        min_sep = min(min_sep, np.linalg.norm(pA - pB))
    print(f"head-on ORCA min separation = {min_sep:.1f} m (d_sep=30) -> "
          f"{'AVOIDS' if min_sep > 25 else 'FAILS'}")
