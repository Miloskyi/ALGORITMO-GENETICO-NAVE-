"""
Motor de Física — Alien Lander v3
FIXES:
  1. Fitness con GUÍA CONTINUA: siempre hay gradiente hacia el pad
  2. Bonus de proximidad progresivo (no cliff edge)
  3. Penalización de velocidad RELATIVA a la distancia (lento lejos = ok, lento cerca = muy ok)
  4. Sin penalización de jerk en física (eso va en GA, no aquí)
  5. Condiciones de aterrizaje más generosas para que el AG aprenda primero
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple

# ── Constantes ────────────────────────────────────────────────────────────────
SCREEN_W   = 1150
SCREEN_H   = 700
GRAVITY    = 0.18
THRUST_MAX = 0.62       # Ligeramente más potencia para mayor maniobrabilidad

LAND_VX_MAX  = 2.8      # Un poco más generoso para que aprenda primero
LAND_VY_MAX  = 3.8
LAND_ANG_MAX = 22.0

PAD_W   = 130           # Plataforma más ancha = más fácil de encontrar
PAD_H   = 14
SPAWN_Y = 80


@dataclass
class LanderState:
    x:      float
    y:      float
    vx:     float = 0.0
    vy:     float = 0.0
    angle:  float = 0.0
    alive:  bool  = True
    landed: bool  = False
    crashed:bool  = False
    t:      int   = 0
    # (x, y, angle_fisico, thrust) — ángulo suavizado para visualización
    trail:  List[Tuple[float,float,float,float]] = field(default_factory=list)

    def record(self, angle: float, thrust: float):
        if len(self.trail) < 500:
            self.trail.append((self.x, self.y, angle, thrust))

    @property
    def out_of_bounds(self) -> bool:
        return self.x < -30 or self.x > SCREEN_W+30 or self.y > SCREEN_H+30


@dataclass
class Platform:
    x:     float
    y:     float = SCREEN_H - 60
    width: int   = PAD_W

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def top_y(self) -> float:
        return self.y


def simulate_lander(chromosome: np.ndarray, platform: Platform, spawn_x: float) -> LanderState:
    state = LanderState(x=spawn_x, y=float(SPAWN_Y))

    for t in range(len(chromosome)):
        if not state.alive:
            break

        raw_angle  = float(chromosome[t, 0])
        raw_thrust = float(chromosome[t, 1])

        # Inercia de rotación
        state.angle += (raw_angle - state.angle) * 0.10
        state.angle  = float(np.clip(state.angle, -45, 45))

        angle_rad = math.radians(state.angle)
        ax = math.sin(angle_rad) * raw_thrust * THRUST_MAX
        ay = -math.cos(angle_rad) * raw_thrust * THRUST_MAX + GRAVITY

        state.vx += ax
        state.vy += ay
        state.vx  = float(np.clip(state.vx, -14.0, 14.0))
        state.vy  = float(np.clip(state.vy, -14.0, 14.0))
        state.x  += state.vx
        state.y  += state.vy
        state.t   = t
        state.record(state.angle, raw_thrust)

        # Colisión con plataforma (zona generosa)
        pad_l = platform.x - 10
        pad_r = platform.x + PAD_W + 10
        pad_t = platform.top_y

        if pad_l < state.x < pad_r and pad_t - 6 < state.y < pad_t + 20:
            on_pad  = abs(state.x - platform.cx) < PAD_W / 2 + 5
            soft    = abs(state.vy) <= LAND_VY_MAX and abs(state.vx) <= LAND_VX_MAX
            upright = abs(state.angle) <= LAND_ANG_MAX
            state.landed  = on_pad and soft and upright
            state.crashed = not state.landed
            state.alive   = False
            state.y = pad_t
            break

        if state.y >= SCREEN_H - 42:
            state.crashed = True
            state.alive   = False
            state.y = SCREEN_H - 42
            break

        if state.out_of_bounds:
            state.crashed = True
            state.alive   = False
            break

    return state


# ── Función de Fitness ────────────────────────────────────────────────────────
# DISEÑO CLAVE: el fitness debe tener gradiente en TODAS las situaciones,
# nunca debe ser plano. Usamos distancia euclidiana normalizada como base.

def compute_fitness(state: LanderState, platform: Platform) -> float:
    dx = state.x - platform.cx
    dy = state.y - platform.top_y
    dist = math.sqrt(dx*dx + dy*dy)

    speed = math.sqrt(state.vx**2 + state.vy**2)
    tilt  = abs(state.angle)

    # ── 1. Guía posicional — SIEMPRE hay gradiente hacia el pad ──────────────
    # Escala: 0 a 2000 puntos por cercanía
    max_dist = math.sqrt(SCREEN_W**2 + SCREEN_H**2)
    pos_score = 2000.0 * (1.0 - dist / max_dist)

    # ── 2. Bonus de alineación horizontal (más importante que la vertical) ───
    # Si está directamente encima del pad, premio extra
    horiz_score = max(0.0, 800.0 * (1.0 - abs(dx) / (SCREEN_W / 2)))

    # ── 3. Penalización de velocidad ESCALADA por distancia ──────────────────
    # Lejos del pad: velocidad alta es ok (está viajando)
    # Cerca del pad: velocidad alta es muy mala
    proximity = max(0.0, 1.0 - dist / 400.0)   # 0=lejos, 1=muy cerca
    speed_pen = speed * (10.0 + 80.0 * proximity)

    # ── 4. Penalización de inclinación (suave) ───────────────────────────────
    tilt_pen = tilt * 2.0

    # ── 5. Bonus de progreso vertical (anima a descender) ────────────────────
    progress = max(0.0, (state.y - SPAWN_Y) / max(1.0, platform.top_y - SPAWN_Y))
    progress_score = 400.0 * progress

    fitness = pos_score + horiz_score + progress_score - speed_pen - tilt_pen

    # ── 6. Premio de aterrizaje ───────────────────────────────────────────────
    if state.landed:
        # Premio base grandísimo + bonus por calidad
        fitness += 10000.0
        fitness -= speed * 60.0
        fitness -= tilt  * 30.0
        # Bonus por aterrizaje muy suave
        if abs(state.vy) < 1.5 and abs(state.vx) < 1.0:
            fitness += 2000.0
    elif state.crashed:
        # Penalización suave — no queremos que el AG evite el pad por miedo
        crash_pen = 200.0 + max(0.0, abs(dx) - PAD_W/2) * 3.0
        fitness  -= crash_pen
    else:
        # Bonos por estar CERCA cuando se acaba el tiempo (sin chocar)
        if dist < 100:
            fitness += 500.0
        if dist < 50:
            fitness += 1000.0

    return float(fitness)


def evaluate_population(population, platform: Platform, spawn_x: float):
    for ind in population:
        state = simulate_lander(ind.chromosome, platform, spawn_x)
        ind.fitness     = compute_fitness(state, platform)
        ind.landed      = state.landed
        ind.crashed     = state.crashed
        ind.final_vx    = state.vx
        ind.final_vy    = state.vy
        ind.final_angle = state.angle
        ind.dist_to_pad = abs(state.x - platform.cx)
    return population