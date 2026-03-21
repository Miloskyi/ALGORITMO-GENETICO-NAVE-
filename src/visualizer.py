"""
Visualizador Pygame — Alien Lander v3
Muestra indicador de estancamiento y diversidad inyectada.
"""

import pygame
import sys
import math
import random
from typing import List

from physics import (
    simulate_lander, compute_fitness, evaluate_population,
    LanderState, Platform,
    SCREEN_W, SCREEN_H, PAD_W, PAD_H, SPAWN_Y,
)
from genetic_algorithm import GeneticAlgorithm, Individual

# ── Paleta ────────────────────────────────────────────────────────────────────
BG_TOP    = (4,   8,  20)
BG_BOT    = (8,  14,  35)
GROUND_C  = (28,  38,  65)
PAD_C     = (40, 200, 150)
PAD_GLOW  = (20, 255, 170)
SHIP_BEST = (255, 215,  50)
SHIP_ALV  = (80,  160, 255)
SHIP_DEAD = (40,   50,  80)
TRAIL_BEST= (80, 160, 255)
TRAIL_ALV = (40,  70, 130)
TEXT_C    = (180, 210, 245)
DIM_C     = (60,   80, 120)
ACCENT    = (80,  140, 255)
GREEN_C   = (50,  220, 140)
RED_C     = (255,  65,  65)
ORANGE_C  = (255, 160,  30)
PANEL_BG  = (8,   12,  28)

FPS     = 60
PANEL_W = 285
TOTAL_W = SCREEN_W + PANEL_W


class StarField:
    def __init__(self):
        self.stars = [(random.randint(0, SCREEN_W), random.randint(0, SCREEN_H),
                       random.random(), random.uniform(0.4, 2.2)) for _ in range(200)]

    def draw(self, surf):
        t = pygame.time.get_ticks() / 1000
        for x, y, ph, sz in self.stars:
            br = int(150 + 80 * math.sin(t * 1.1 + ph * 6.28))
            pygame.draw.circle(surf, (br, br, 255), (x, y), max(1, int(sz / 2)))


def draw_ship(surf, x, y, angle, thrust, color):
    cx, cy = int(x), int(y)
    s = pygame.Surface((72, 82), pygame.SRCALPHA)
    c = color
    pygame.draw.ellipse(s, (*c, 210), (8, 22, 52, 28))
    pygame.draw.ellipse(s, (*c, 160), (16,  6, 38, 28))
    pygame.draw.ellipse(s, (160, 235, 255, 200), (22, 10, 26, 18))
    lc = (max(0,c[0]-50), max(0,c[1]-50), max(0,c[2]-50), 190)
    pygame.draw.line(s, lc, (18, 48), ( 6, 64), 3)
    pygame.draw.line(s, lc, (52, 48), (64, 64), 3)
    pygame.draw.line(s, lc, (35, 52), (35, 68), 3)
    if (pygame.time.get_ticks() // 500) % 2:
        pygame.draw.circle(s, (255, 40, 40, 220), (12, 34), 4)
        pygame.draw.circle(s, ( 40,255, 90, 220), (58, 34), 4)
    if thrust > 0.05:
        fl = int(thrust * 30 + 8)
        for i, fc in enumerate([(255,200,50),(255,110,20),(200,50,10)]):
            w = max(1, int((3-i)*2.5*thrust))
            pygame.draw.line(s, (*fc, 200-i*55), (35, 52), (35, 52+fl), w)
    rot = pygame.transform.rotate(s, -angle)
    surf.blit(rot, rot.get_rect(center=(cx, cy)))


def draw_trail(surf, trail, color, max_alpha=160):
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        a = int(max_alpha * i / len(trail))
        c = (int(color[0]*a//max_alpha), int(color[1]*a//max_alpha), int(color[2]*a//max_alpha))
        pygame.draw.line(surf, c,
                         (int(trail[i-1][0]), int(trail[i-1][1])),
                         (int(trail[i][0]),   int(trail[i][1])), 1)


def draw_platform(surf, platform: Platform):
    t = pygame.time.get_ticks()
    px, py = int(platform.x), int(platform.y)
    glow_a = int(100 + 70 * math.sin(t / 500))
    gs = pygame.Surface((PAD_W + 24, 22), pygame.SRCALPHA)
    pygame.draw.rect(gs, (*PAD_GLOW, glow_a), (0, 0, PAD_W + 24, 22), border_radius=6)
    surf.blit(gs, (px - 12, py - 6))
    pygame.draw.rect(surf, PAD_C,   (px, py, PAD_W, PAD_H), border_radius=3)
    pygame.draw.rect(surf, PAD_GLOW,(px+6, py+3, PAD_W-12, 5), border_radius=2)
    pygame.draw.line(surf, (*PAD_GLOW, 120),
                     (int(platform.cx), py - 25),
                     (int(platform.cx), py), 1)


def draw_graph(surf, rect, h_best, h_avg, font):
    gx, gy, gw, gh = rect
    pygame.draw.rect(surf, (6, 10, 24), (gx, gy, gw, gh), border_radius=4)
    pygame.draw.rect(surf, DIM_C,       (gx, gy, gw, gh), 1, border_radius=4)
    if len(h_best) < 2:
        return
    maxv = max(max(h_best), 1)
    minv = min(min(h_best), 0)
    rng  = maxv - minv or 1
    def px(i): return gx + int(i / max(len(h_best)-1,1) * (gw-6)) + 3
    def py(v): return gy + gh - 4 - int((v-minv)/rng*(gh-10))
    if len(h_avg) >= 2:
        pts = [(px(i), py(v)) for i,v in enumerate(h_avg)]
        pygame.draw.lines(surf, GREEN_C, False, pts, 1)
    pts = [(px(i), py(v)) for i,v in enumerate(h_best)]
    pygame.draw.lines(surf, ACCENT, False, pts, 2)
    surf.blit(font.render("FITNESS", True, DIM_C), (gx+4, gy+3))


class LandingVisualizer:
    def __init__(self, ga: GeneticAlgorithm, platform: Platform,
                 spawn_x: float, speed: int = 2):
        pygame.init()
        self.window   = pygame.display.set_mode((TOTAL_W, SCREEN_H))
        pygame.display.set_caption("ALIEN LANDER GA v3")
        self.font_big = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_mid = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_sm  = pygame.font.SysFont("monospace", 11)
        self.clock    = pygame.time.Clock()
        self.stars    = StarField()
        self.ga       = ga
        self.platform = platform
        self.spawn_x  = spawn_x
        self.speed    = speed
        self.running  = True

        # ── Estado en vivo del panel — se actualiza en tiempo real ─────────────
        self._live_gen         = 0
        self._live_best_fit    = 0.0
        self._live_avg_fit     = 0.0
        self._live_best_ever   = 0.0
        self._live_landed      = 0
        self._live_stagnation  = 0
        self._live_best_state: LanderState = None   # estado final del mejor de esta gen
        self._live_best_ind: Individual    = None   # individuo del mejor de esta gen
        self._history_best: list           = []
        self._history_avg:  list           = []

    def run(self):
        while self.running:
            self._run_generation()
            if not self.running:
                break
            self.ga.evolve()
            # Sincronizar historial desde el GA después de evolucionar
            self._history_best = self.ga.history_best[:]
            self._history_avg  = self.ga.history_avg[:]
        pygame.quit()

    def _run_generation(self):
        pop    = self.ga.population
        # Simular toda la generación y calcular fitness
        states = [simulate_lander(ind.chromosome, self.platform, self.spawn_x) for ind in pop]
        for i, ind in enumerate(pop):
            ind.fitness     = compute_fitness(states[i], self.platform)
            ind.landed      = states[i].landed
            ind.crashed     = states[i].crashed
            ind.final_vx    = states[i].vx
            ind.final_vy    = states[i].vy
            ind.final_angle = states[i].angle
            ind.dist_to_pad = abs(states[i].x - self.platform.cx)

        # Calcular stats en vivo AHORA que el fitness está asignado
        fits      = [ind.fitness for ind in pop]
        best_idx  = int(max(range(len(pop)), key=lambda i: fits[i]))
        best_fit  = fits[best_idx]
        avg_fit   = sum(fits) / len(fits)
        landed_n  = sum(1 for ind in pop if ind.landed)

        # Actualizar mejor histórico en vivo
        if best_fit > self._live_best_ever:
            self._live_best_ever = best_fit

        # Guardar estado en vivo para el panel
        self._live_gen        = self.ga.generation
        self._live_best_fit   = best_fit
        self._live_avg_fit    = avg_fit
        self._live_landed     = landed_n
        self._live_stagnation = self.ga._stagnation_counter
        self._live_best_state = states[best_idx]
        self._live_best_ind   = pop[best_idx]

        max_steps = max((s.t for s in states), default=1)
        step = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False; return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.speed = min(self.speed + 1, 12)
                    if event.key == pygame.K_DOWN:
                        self.speed = max(self.speed - 1, 1)
                    if event.key == pygame.K_s:
                        self.ga.save("best_lander.json")

            for _ in range(self.speed):
                step += 1

            self._render(states, best_idx, step)

            if step >= max_steps + 60:
                break

    def _render(self, states: List[LanderState], best_idx: int, step: int):
        surf = self.window

        # Fondo degradado
        for row in range(0, SCREEN_H, 4):
            r = int(BG_TOP[0] + (BG_BOT[0]-BG_TOP[0]) * row/SCREEN_H)
            g = int(BG_TOP[1] + (BG_BOT[1]-BG_TOP[1]) * row/SCREEN_H)
            b = int(BG_TOP[2] + (BG_BOT[2]-BG_TOP[2]) * row/SCREEN_H)
            pygame.draw.rect(surf, (r,g,b), (0, row, SCREEN_W, 4))

        self.stars.draw(surf)
        pygame.draw.rect(surf, GROUND_C, (0, SCREEN_H-40, SCREEN_W, 40))
        pygame.draw.line(surf, (45, 60, 100), (0, SCREEN_H-40), (SCREEN_W, SCREEN_H-40), 2)
        draw_platform(surf, self.platform)

        # Trails (no-mejor)
        for i, s in enumerate(states):
            if i == best_idx: continue
            end = min(step, len(s.trail))
            draw_trail(surf, [(p[0], p[1]) for p in s.trail[:end]], TRAIL_ALV, 70)

        # Trail del mejor
        s_best = states[best_idx]
        draw_trail(surf,
                   [(p[0], p[1]) for p in s_best.trail[:min(step, len(s_best.trail))]],
                   TRAIL_BEST, 180)

        # Naves no-mejores
        for i, s in enumerate(states):
            if i == best_idx: continue
            end = min(step, len(s.trail))
            if end < 1: continue
            sx, sy, ang, thr = s.trail[end-1]
            col = SHIP_DEAD if (s.landed or s.crashed) else SHIP_ALV
            draw_ship(surf, sx, sy, ang, thr if not (s.landed or s.crashed) else 0, col)

        # Mejor nave (amarilla)
        end = min(step, len(s_best.trail))
        if end >= 1:
            sx, sy, ang, thr = s_best.trail[end-1]
            draw_ship(surf, sx, sy, ang, thr if s_best.alive else 0, SHIP_BEST)

        # HUD superior
        hud = self.font_mid.render(
            f"Gen {self._live_gen}  x{self.speed} vel  [UP/DOWN] velocidad  [S] guardar",
            True, DIM_C)
        surf.blit(hud, (8, 8))

        stag = self._live_stagnation
        if stag > 3:
            stag_txt = self.font_sm.render(
                f"INYECTANDO DIVERSIDAD (estancado {stag} gen)", True, ORANGE_C)
            surf.blit(stag_txt, (8, 26))

        if self._live_landed > 0:
            ok = self.font_mid.render(f"ATERRIZO: {self._live_landed}", True, GREEN_C)
            surf.blit(ok, (8, 44))

        self._draw_panel()
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_panel(self):
        panel = pygame.Surface((PANEL_W, SCREEN_H))
        panel.fill(PANEL_BG)
        pygame.draw.line(panel, DIM_C, (0, 0), (0, SCREEN_H), 2)

        y = 12
        def txt(t, col=TEXT_C, font=None):
            nonlocal y
            f = font or self.font_mid
            # Renderizar caracter a caracter para evitar problemas con emojis
            s = f.render(str(t), True, col)
            panel.blit(s, (12, y))
            y += s.get_height() + 4

        def sep():
            nonlocal y
            pygame.draw.line(panel, DIM_C, (8, y), (PANEL_W-8, y))
            y += 6

        def kv(label, value, vcol=TEXT_C):
            """Key-value con label dim y valor alineado a la derecha."""
            nonlocal y
            lbl_s = self.font_sm.render(label, True, DIM_C)
            val_s = self.font_sm.render(str(value), True, vcol)
            panel.blit(lbl_s, (12, y))
            panel.blit(val_s, (PANEL_W - val_s.get_width() - 12, y))
            y += lbl_s.get_height() + 5
            pygame.draw.line(panel, (20, 28, 50), (12, y-1), (PANEL_W-12, y-1))

        # ── Título ──
        txt("ALIEN LANDER GA v3", SHIP_BEST, self.font_big)
        sep()

        # ── Evolución ──
        txt("EVOLUCION", DIM_C, self.font_sm)
        y += 2
        kv("Generacion",    self._live_gen,                    ACCENT)
        kv("Aterrizados",   f"{self._live_landed}/{self.ga.population_size}", GREEN_C)
        stag = self._live_stagnation
        if stag > 0:
            kv("Estancamiento", f"{stag} gen",
               ORANGE_C if stag > 4 else DIM_C)
        sep()

        # ── Fitness ──
        txt("FITNESS", DIM_C, self.font_sm)
        y += 2
        kv("Mejor (gen actual)",  f"{self._live_best_fit:>8.0f}",  TEXT_C)
        kv("Promedio (gen act.)", f"{self._live_avg_fit:>8.0f}",   TEXT_C)
        kv("Record historico",   f"{self._live_best_ever:>8.0f}", SHIP_BEST)
        sep()

        # ── Mejor individuo de la generación actual ──
        txt("MEJOR INDIVIDUO", DIM_C, self.font_sm)
        y += 2
        ind = self._live_best_ind
        st  = self._live_best_state
        if ind is not None and st is not None:
            estado = ("ATERRIZO" if ind.landed
                      else ("CHOCO" if ind.crashed else "EN VUELO"))
            ecol = (GREEN_C if ind.landed else RED_C if ind.crashed else ACCENT)
            kv("Estado",    estado,                      ecol)
            kv("Vel X",     f"{ind.final_vx:>+7.2f} px/f", TEXT_C)
            kv("Vel Y",     f"{ind.final_vy:>+7.2f} px/f", TEXT_C)
            kv("Angulo",    f"{ind.final_angle:>+6.1f} deg", TEXT_C)
            kv("Dist pad",  f"{ind.dist_to_pad:>7.1f} px",  TEXT_C)
        else:
            txt("-- calculando --", DIM_C, self.font_sm)
        sep()

        # ── Parámetros ──
        txt("PARAMETROS AG", DIM_C, self.font_sm)
        y += 2
        kv("Poblacion",    self.ga.population_size,  TEXT_C)
        kv("Cromosoma",    "200 x 2 genes",          TEXT_C)
        kv("Mutacion",     f"{self.ga.mutation_rate:.0%} (adapt.)", TEXT_C)
        kv("Cruce",        "BLX-a + 2pt  85%",       TEXT_C)
        kv("Elitismo",     self.ga.elitism,           TEXT_C)
        kv("Init sesgada", "65%",                     TEXT_C)
        kv("Pad X / Y",    f"{int(self.platform.cx)} / {int(self.platform.top_y)}", DIM_C)
        sep()

        # ── Gráfico ──
        gh = min(120, SCREEN_H - y - 18)
        if gh > 35 and len(self._history_best) >= 2:
            draw_graph(panel, (8, y, PANEL_W-16, gh),
                       self._history_best, self._history_avg, self.font_sm)
            y += gh + 5
            b_s = self.font_sm.render("  Mejor", True, ACCENT)
            a_s = self.font_sm.render("  Prom",  True, GREEN_C)
            pygame.draw.line(panel, ACCENT,   (12,      y+5), (28,      y+5), 2)
            pygame.draw.line(panel, GREEN_C,  (12+60,   y+5), (28+60,   y+5), 1)
            panel.blit(b_s, (28,    y))
            panel.blit(a_s, (28+60, y))
            y += 16
        elif gh > 35:
            txt("grafico disponible", DIM_C, self.font_sm)
            txt("desde gen 2", DIM_C, self.font_sm)

        self.window.blit(panel, (SCREEN_W, 0))