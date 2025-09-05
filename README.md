# Curvature-Based Vessel Motion Modeling  
*A Kinematic Approach for Real-Time Path Planning and Predictive Navigation*

[![Conference](https://img.shields.io/badge/Conference-AYO%20Colloquium%202025-blue)](https://)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

---

## About This Work  

This repository contains code, examples, and supplementary material for the article:  

**Curvature-Based Vessel Motion Modeling: A Kinematic Approach for Real-Time Path Planning and Predictive Navigation**  

---

## Motivation  

Path planning for **Unmanned Surface Vessels (USVs)** is fundamentally different from ground robots or aerial drones.  
- USVs cannot perform **sharp waypoint transitions** due to hydrodynamic constraints.  
- The **minimum turning radius** grows with speed and is bounded by rudder angle limits.  
- Classical shortest-path planners (straight segments, waypoints) produce **dynamically infeasible trajectories**.  

This project introduces a **curvature-based kinematic framework** that reformulates vessel motion in terms of:  
- **Curvature (κ)** – directly linked to rudder angle and maneuvering capability.  
- **Arc length (s)** – progression along the path, linked to surge speed.  

This formulation naturally enforces turning radius limits and produces **smooth, feasible, and controller-friendly paths**.  

---

## Methodology  

The methodology has three main pillars:  

### 1. Curvilinear Kinematic Formulation  
We reformulate the vessel dynamics in arc-length coordinates:  

\[
\frac{dx}{ds} = \cos(\chi(s)), \quad 
\frac{dy}{ds} = \sin(\chi(s)), \quad 
\frac{d\chi}{ds} = \kappa(s)
\]

- \( \chi \): course angle  
- \( \kappa(s) \): curvature profile along the path  
- This representation is **vehicle-agnostic** and compact, relying only on steady-state turning data.  

---

### 2. Sinc-Based Discretization  
A new discretization scheme is applied for numerical robustness:  

\[
x_{k+1} = x_k + \Delta s \, \text{sinc}\!\left(\frac{\Delta \chi}{2}\right) \cos\!\left(\chi_k + \frac{\Delta \chi}{2}\right)
\]  

\[
y_{k+1} = y_k + \Delta s \, \text{sinc}\!\left(\frac{\Delta \chi}{2}\right) \sin\!\left(\chi_k + \frac{\Delta \chi}{2}\right)
\]

- **Unifies straight and curved motion** in a single update.  
- Eliminates singularities as curvature → 0.  
- Provides smooth numerical integration across path segments.  

---

### 3. Hierarchical Planning Architecture  

The planning pipeline has **two levels**:  

1. **Global Planner**  
   - Builds a **visibility graph** over quadtree-sampled free space.  
   - Uses A* search to generate a collision-free, shortest waypoint chain.  
   - Fast, but curvature-agnostic.  

2. **Local Refinement (Offline NMPC)**  
   - Optimizes curvature \( \kappa_k \) and arc-length steps \( \Delta s_k \).  
   - Enforces:  
     - Speed-dependent turning radius limits.  
     - Obstacle clearance.  
     - Continuity of curvature across waypoints.  
   - Solved with **CasADi + IPOPT**.  
   - Produces **smooth, dynamically feasible reference trajectories**.  

---

## Key Contributions  

- **Curvature-aware kinematics**: A closed-form mapping from curvature profiles to Cartesian paths.  
- **Numerically robust discretization**: Sinc formulation ensuring smooth transition between straight and curved motion.  
- **Hierarchical planning**:  
  - Global collision-free routes via quadtree-based visibility graphs.  
  - Local refinement with NMPC for curvature continuity and dynamic feasibility.  
- **Simulation-validated**: Realistic obstacle scenarios demonstrate feasibility, smoothness, and real-time suitability.  

---

## Results  

- Sharp waypoint turns are replaced with **feasible arcs** that respect vessel dynamics.  
- Adjustable turning radius constraint shows trade-off between path length and maneuverability.  
- Obstacle-rich environments are handled by combining **global visibility graphs** with **curvature-constrained refinement**.  
- Computation time per NMPC step: **20–30 ms** on Intel i7-11800H — suitable for real-time.  

---

## Repository Structure  

