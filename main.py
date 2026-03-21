"""
🛸 Alien Lander GA v3
La plataforma es aleatoria al arrancar. El AG usa esa info para
inicializar la población de forma SESGADA hacia el pad.

Uso:
    python main.py                     # Con visualización Pygame
    python main.py --headless          # Sin visualización
    python main.py --headless --generations 300
    python main.py --population 60 --speed 3
"""

import argparse
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from genetic_algorithm import GeneticAlgorithm
from physics import (
    Platform, evaluate_population,
    SCREEN_W, SCREEN_H, PAD_W, SPAWN_Y,
)


def make_random_platform() -> Platform:
    pad_x = random.uniform(100, SCREEN_W - 100 - PAD_W)
    pad_y = random.uniform(SCREEN_H * 0.55, SCREEN_H - 65)
    return Platform(x=pad_x, y=pad_y)


def make_random_spawn(platform: Platform) -> float:
    while True:
        sx = random.uniform(120, SCREEN_W - 120)
        if abs(sx - platform.cx) > 180:
            return sx


def train_headless(ga: GeneticAlgorithm, platform: Platform,
                   spawn_x: float, generations: int):
    print(f"""
╔═══════════════════════════════════════════════════╗
║   🛸  ALIEN LANDER GA v3 — Modo Headless          ║
╠═══════════════════════════════════════════════════╣
║  Plataforma  X: {int(platform.cx):<33}║
║  Plataforma  Y: {int(platform.top_y):<33}║
║  Spawn nave  X: {int(spawn_x):<33}║
║  Población    : {ga.population_size:<33}║
╚═══════════════════════════════════════════════════╝
""")

    for gen in range(generations):
        evaluate_population(ga.population, platform, spawn_x)
        stats  = ga.get_stats()
        stag   = f" [estancado {stats['stagnation']}]" if stats['stagnation'] > 3 else ""
        mark   = " ✓ ¡ATERRIZÓ!" if stats["landed"] else ""
        print(
            f"Gen {stats['generation']:>4} │ "
            f"Mejor: {stats['best_fitness']:>8.1f} │ "
            f"Prom: {stats['avg_fitness']:>7.1f} │ "
            f"Récord: {stats['best_ever']:>8.1f} │ "
            f"Aterrizados: {stats['landed']}{mark}{stag}"
        )
        ga.evolve()

    ga.save("best_lander.json")
    print("\n✅ Listo. Guardado en best_lander.json")


def main():
    parser = argparse.ArgumentParser(description="🛸 Alien Lander GA v3")
    parser.add_argument("--headless",    action="store_true")
    parser.add_argument("--generations", type=int,   default=300)
    parser.add_argument("--population",  type=int,   default=50)
    parser.add_argument("--mutation",    type=float, default=0.04)
    parser.add_argument("--speed",       type=int,   default=2)
    args = parser.parse_args()

    # Plataforma aleatoria FIJA por ejecución
    platform = make_random_platform()
    spawn_x  = make_random_spawn(platform)

    print(f"""
╔═══════════════════════════════════════════════════╗
║       🛸  ALIEN LANDER — Algoritmo Genético v3    ║
╠═══════════════════════════════════════════════════╣
║  Cromosoma  : 200 pasos × (ángulo, potencia)      ║
║  Plataforma : X={int(platform.cx):<6} Y={int(platform.top_y):<23}║
║  Spawn      : X={int(spawn_x):<34}║
║  Población  : {args.population:<36}║
║  Init sesgada hacia el pad: 65%                   ║
╚═══════════════════════════════════════════════════╝
""")

    # GA recibe la posición del pad para inicialización sesgada
    ga = GeneticAlgorithm(
        population_size=args.population,
        mutation_rate=args.mutation,
        elitism=6,
        crossover_rate=0.85,
        platform_cx=platform.cx,
        spawn_x=spawn_x,
        screen_w=SCREEN_W,
        screen_h=SCREEN_H,
        pad_y=platform.top_y,
    )

    if args.headless:
        train_headless(ga, platform, spawn_x, args.generations)
    else:
        try:
            from visualizer import LandingVisualizer
            viz = LandingVisualizer(ga, platform, spawn_x, speed=args.speed)
            viz.run()
        except ImportError:
            print("⚠  Pygame no encontrado. Ejecutando modo headless...")
            train_headless(ga, platform, spawn_x, args.generations)


if __name__ == "__main__":
    main()