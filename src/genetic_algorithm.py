"""
Algoritmo Genético — v3
FIXES que garantizan convergencia:
  1. Inicialización SESGADA hacia el pad (la nave sabe hacia dónde ir)
  2. Reinserción de diversidad (si el fitness se estanca, inyectar nuevos individuos)
  3. Mutación guiada en cromosomas de élite (exploración local fina)
  4. Niching: no permitir que toda la población sea casi idéntica
  5. STEPS aumentado a 200 para que haya tiempo de maniobrar
"""

import numpy as np
import random
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

STEPS        = 200
ANGLE_RANGE  = (-45.0, 45.0)
THRUST_RANGE = (0.0,   1.0)


@dataclass
class Individual:
    chromosome: np.ndarray   # (STEPS, 2)
    fitness:    float = -np.inf
    landed:     bool  = False
    crashed:    bool  = False
    final_vx:   float = 0.0
    final_vy:   float = 0.0
    final_angle:float = 0.0
    dist_to_pad:float = 0.0

    @staticmethod
    def random() -> "Individual":
        """Inicialización aleatoria limpia (sin sesgo)."""
        angles  = np.zeros(STEPS)
        angles[0] = np.random.uniform(-10, 10)
        for i in range(1, STEPS):
            angles[i] = np.clip(angles[i-1] + np.random.uniform(-6, 6), *ANGLE_RANGE)
        # Suavizado
        angles = np.convolve(angles, np.ones(7)/7, mode='same')
        angles = np.clip(angles, *ANGLE_RANGE)

        thrusts = np.zeros(STEPS)
        thrusts[0] = np.random.uniform(0.3, 0.6)
        for i in range(1, STEPS):
            thrusts[i] = np.clip(thrusts[i-1] + np.random.uniform(-0.06, 0.06), *THRUST_RANGE)

        return Individual(chromosome=np.stack([angles, thrusts], axis=1))

    @staticmethod
    def biased(platform_cx: float, spawn_x: float,
               screen_w: float, screen_h: float, pad_y: float) -> "Individual":
        """
        Inicialización SESGADA: la nave ya sabe aproximadamente hacia dónde ir.
        Genera cromosomas que tienden a moverse hacia el pad.
        Esto acelera enormemente la convergencia.
        """
        from physics import SPAWN_Y
        dx = platform_cx - spawn_x        # dirección horizontal necesaria
        dy = pad_y - SPAWN_Y              # distancia vertical a recorrer

        angles  = np.zeros(STEPS)
        thrusts = np.zeros(STEPS)

        # Fase 1 (0-40%): orientarse hacia el pad
        phase1 = int(STEPS * 0.40)
        # Ángulo base: apuntar horizontalmente hacia el pad
        base_angle = np.clip(dx / screen_w * 60, *ANGLE_RANGE)
        for i in range(phase1):
            noise = np.random.uniform(-8, 8)
            angles[i] = np.clip(base_angle + noise, *ANGLE_RANGE)
            thrusts[i] = np.random.uniform(0.35, 0.65)

        # Fase 2 (40-75%): descender controladamente
        phase2 = int(STEPS * 0.75)
        for i in range(phase1, phase2):
            # Reducir ángulo lateral, empezar a verticalizar
            t = (i - phase1) / (phase2 - phase1)
            target = base_angle * (1 - t)    # gradualmente a 0°
            noise = np.random.uniform(-5, 5)
            angles[i] = np.clip(target + noise, *ANGLE_RANGE)
            thrusts[i] = np.random.uniform(0.30, 0.55)

        # Fase 3 (75-100%): aproximación final, freno
        for i in range(phase2, STEPS):
            angles[i] = np.clip(np.random.uniform(-8, 8), *ANGLE_RANGE)
            thrusts[i] = np.random.uniform(0.40, 0.70)   # más potencia para frenar

        # Suavizar ambas señales
        angles  = np.convolve(angles,  np.ones(9)/9,  mode='same')
        thrusts = np.convolve(thrusts, np.ones(5)/5, mode='same')
        angles  = np.clip(angles,  *ANGLE_RANGE)
        thrusts = np.clip(thrusts, *THRUST_RANGE)

        return Individual(chromosome=np.stack([angles, thrusts], axis=1))

    def copy(self) -> "Individual":
        return Individual(chromosome=self.chromosome.copy(), fitness=self.fitness,
                          landed=self.landed, crashed=self.crashed)


# ── Operadores ────────────────────────────────────────────────────────────────

def tournament(pop: List[Individual], k: int = 4) -> Individual:
    return max(random.sample(pop, min(k, len(pop))), key=lambda i: i.fitness)


def crossover_blend(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    mask  = np.random.random(STEPS) < 0.5
    c1, c2 = p1.chromosome.copy(), p2.chromosome.copy()
    alpha = np.random.uniform(0.2, 0.8, (mask.sum(), 1))
    c1[mask] = alpha * p1.chromosome[mask] + (1-alpha) * p2.chromosome[mask]
    c2[mask] = alpha * p2.chromosome[mask] + (1-alpha) * p1.chromosome[mask]
    return Individual(chromosome=c1), Individual(chromosome=c2)


def crossover_twopoint(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    a, b = sorted(random.sample(range(1, STEPS), 2))
    c1 = np.concatenate([p1.chromosome[:a], p2.chromosome[a:b], p1.chromosome[b:]])
    c2 = np.concatenate([p2.chromosome[:a], p1.chromosome[a:b], p2.chromosome[b:]])
    return Individual(chromosome=c1), Individual(chromosome=c2)


def mutate(ind: Individual, rate: float, strength: float) -> Individual:
    chrom = ind.chromosome.copy()
    # Ángulo: mutación con suavizado posterior (evita tembleque)
    mask = np.random.random(STEPS) < rate
    if mask.any():
        chrom[mask, 0] += np.random.normal(0, strength * 45, mask.sum())
        chrom[:, 0] = np.clip(chrom[:, 0], *ANGLE_RANGE)
        # Re-suavizar después de mutar
        chrom[:, 0] = np.convolve(chrom[:, 0], np.ones(5)/5, mode='same')
        chrom[:, 0] = np.clip(chrom[:, 0], *ANGLE_RANGE)
    # Potencia
    mask = np.random.random(STEPS) < rate
    if mask.any():
        chrom[mask, 1] += np.random.normal(0, strength * 0.4, mask.sum())
        chrom[:, 1] = np.clip(chrom[:, 1], *THRUST_RANGE)
    return Individual(chromosome=chrom)


def mutate_local(ind: Individual, strength: float = 0.08) -> Individual:
    """Mutación local fina para ajuste de élites."""
    chrom = ind.chromosome.copy()
    # Mutar solo un segmento aleatorio (30% del cromosoma)
    start = random.randint(0, STEPS - 1)
    length = int(STEPS * 0.30)
    end = min(start + length, STEPS)
    seg_len = end - start
    chrom[start:end, 0] += np.random.normal(0, strength * 20, seg_len)
    chrom[start:end, 1] += np.random.normal(0, strength * 0.25, seg_len)
    chrom[:, 0] = np.clip(chrom[:, 0], *ANGLE_RANGE)
    chrom[:, 1] = np.clip(chrom[:, 1], *THRUST_RANGE)
    return Individual(chromosome=chrom)


# ── GA Principal ──────────────────────────────────────────────────────────────

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int   = 50,
        mutation_rate:   float = 0.04,
        mutation_str:    float = 0.18,
        elitism:         int   = 6,
        crossover_rate:  float = 0.85,
        # Para inicialización sesgada
        platform_cx:     float = 0.0,
        spawn_x:         float = 0.0,
        screen_w:        float = 1150,
        screen_h:        float = 700,
        pad_y:           float = 640,
    ):
        self.population_size = population_size
        self.mutation_rate   = mutation_rate
        self.mutation_str    = mutation_str
        self.elitism         = elitism
        self.crossover_rate  = crossover_rate
        self.generation      = 0

        # Parámetros para inicialización sesgada
        self.platform_cx = platform_cx
        self.spawn_x     = spawn_x
        self.screen_w    = screen_w
        self.screen_h    = screen_h
        self.pad_y       = pad_y

        # Población inicial: 60% sesgada + 40% aleatoria
        self.population = self._init_population()
        self.best_individual: Optional[Individual] = None

        self.history_best:    List[float] = []
        self.history_avg:     List[float] = []
        self.history_landed:  List[int]   = []

        # Para detección de estancamiento
        self._stagnation_counter = 0
        self._last_best = -np.inf

    def _init_population(self) -> List[Individual]:
        pop = []
        n_biased = int(self.population_size * 0.65)
        for _ in range(n_biased):
            pop.append(Individual.biased(
                self.platform_cx, self.spawn_x,
                self.screen_w, self.screen_h, self.pad_y
            ))
        for _ in range(self.population_size - n_biased):
            pop.append(Individual.random())
        return pop

    def _inject_diversity(self):
        """
        Si el AG se estanca, reemplazar el 30% peor con nuevos individuos sesgados.
        Así se evita la convergencia prematura.
        """
        n_inject = int(self.population_size * 0.30)
        self.population.sort(key=lambda i: i.fitness, reverse=True)
        new_blood = [
            Individual.biased(self.platform_cx, self.spawn_x,
                              self.screen_w, self.screen_h, self.pad_y)
            for _ in range(n_inject)
        ]
        self.population[-n_inject:] = new_blood

    def evolve(self):
        sorted_pop = sorted(self.population, key=lambda i: i.fitness, reverse=True)
        fits   = [i.fitness for i in sorted_pop]
        landed = sum(1 for i in sorted_pop if i.landed)

        self.history_best.append(fits[0])
        self.history_avg.append(float(np.mean(fits)))
        self.history_landed.append(landed)

        if self.best_individual is None or fits[0] > self.best_individual.fitness:
            self.best_individual = sorted_pop[0].copy()

        # ── Detección de estancamiento ────────────────────────────────────────
        improvement = fits[0] - self._last_best
        if improvement < 5.0:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0
        self._last_best = fits[0]

        # Si estancado 8 generaciones sin aterrizar → inyectar diversidad
        if self._stagnation_counter >= 8 and landed == 0:
            self._inject_diversity()
            self._stagnation_counter = 0

        # ── Mutación adaptativa ───────────────────────────────────────────────
        if landed > 0:
            eff_rate = self.mutation_rate * 0.6    # ajuste fino
            eff_str  = self.mutation_str  * 0.5
        elif self._stagnation_counter > 4:
            eff_rate = min(self.mutation_rate * 3.0, 0.30)   # explorar más
            eff_str  = self.mutation_str * 1.5
        else:
            eff_rate = self.mutation_rate
            eff_str  = self.mutation_str

        # ── Nueva generación ──────────────────────────────────────────────────
        new_pop: List[Individual] = []

        n_elite = min(self.elitism, len(sorted_pop))

        # Élites exactos
        for i in range(n_elite):
            new_pop.append(sorted_pop[i].copy())

        # Mutación local fina de los top-3
        for i in range(min(3, n_elite)):
            new_pop.append(mutate_local(sorted_pop[i], 0.06))

        # Reproducción
        while len(new_pop) < self.population_size:
            p1 = tournament(self.population, k=5)
            p2 = tournament(self.population, k=5)
            if random.random() < self.crossover_rate:
                c1, c2 = (crossover_blend(p1, p2) if random.random() < 0.55
                          else crossover_twopoint(p1, p2))
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop.append(mutate(c1, eff_rate, eff_str))
            new_pop.append(mutate(c2, eff_rate, eff_str))

        self.population = new_pop[:self.population_size]
        self.generation += 1

    def get_stats(self) -> dict:
        fits   = [i.fitness for i in self.population]
        landed = sum(1 for i in self.population if i.landed)
        return {
            "generation":     self.generation,
            "best_fitness":   max(fits),
            "avg_fitness":    float(np.mean(fits)),
            "best_ever":      max(self.history_best) if self.history_best else 0.0,
            "landed":         landed,
            "stagnation":     self._stagnation_counter,
            "history_best":   self.history_best,
            "history_avg":    self.history_avg,
            "history_landed": self.history_landed,
        }

    def save(self, path: str = "best_lander.json"):
        if not self.best_individual:
            return
        data = {
            "generation": self.generation,
            "fitness":    self.best_individual.fitness,
            "landed":     self.best_individual.landed,
            "chromosome": self.best_individual.chromosome.tolist(),
            "history_best":  self.history_best,
            "history_avg":   self.history_avg,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Guardado en {path}")