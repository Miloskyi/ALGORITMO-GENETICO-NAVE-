# 🛸 Alien Lander GA — Aterrizaje con Algoritmo Genético

> Algoritmo Genético que optimiza la secuencia de **ángulo y potencia** de empuje
> para aterrizar una nave alienígena sobre una plataforma móvil con combustible limitado.

---

## 📁 Estructura del Proyecto

```
alien_landing/
├── main.py                    # Punto de entrada
├── requirements.txt
├── best_lander.json           # Se genera al guardar (tecla S)
│
├── src/
│   ├── genetic_algorithm.py  # Núcleo AG: selección, cruce, mutación
│   ├── physics.py            # Física: gravedad, empuje, colisiones, fitness
│   └── visualizer.py         # Visualización Pygame
│
└── web/
    └── index.html            # Dashboard web (abrir en navegador)
```

---

## 🧬 Diseño del Algoritmo Genético

### Problema de Optimización
**Minimizar**: distancia al pad + velocidad de impacto + consumo de combustible  
**Maximizar**: precisión de aterrizaje + combustible restante

### Representación del Cromosoma

Cada individuo es un vector de **240 genes reales** (120 pasos × 2 valores):

```
Cromosoma = [ (ángulo₀, potencia₀), (ángulo₁, potencia₁), ..., (ángulo₁₁₉, potencia₁₁₉) ]
             ángulo  ∈ [-45°, 45°]
             potencia ∈ [0.0, 1.0]
```

### Función de Fitness (multi-objetivo)

```python
fitness = 1000 - dist * 1.5          # Proximidad a la plataforma
        - speed_impact * 40           # Penalizar velocidad al aterrizar
        - abs(angle) * 3              # Penalizar inclinación
        + fuel_left * 150             # Premio por combustible restante
        + 2000 (si aterrizó)          # Premio principal
        + fuel_left * 300 (si aterrizó)
        - 500 (si chocó)
        + progress * 100              # Progreso vertical
```

### Operadores Genéticos

| Operador | Técnica | Descripción |
|----------|---------|-------------|
| **Selección** | Torneo (k=4) | 4 candidatos aleatorios, gana el de mayor fitness |
| **Cruce** | BLX-α (blend) / 2 puntos | Interpolación ponderada entre padres |
| **Mutación** | Gaussiana adaptativa | Ruido N(0, σ), σ aumenta si nadie aterriza |
| **Elitismo** | Top 4 | Los 4 mejores pasan intactos |

### Adaptación Dinámica
Si ningún individuo aterriza en una generación, la tasa de mutación se duplica automáticamente para explorar más el espacio de soluciones.

---

## 🚀 Instalación y Uso

```bash
# Instalar dependencias
pip install -r requirements.txt

# Con visualización Pygame (recomendado)
python main.py

# Sin visualización (mucho más rápido)
python main.py --headless --generations 200

# Opciones avanzadas
python main.py --population 60 --mutation 0.08 --speed 3
```

### Controles en Pygame
| Tecla | Acción |
|-------|--------|
| `↑` | Aumentar velocidad de simulación |
| `↓` | Reducir velocidad de simulación |
| `S` | Guardar el mejor individuo en `best_lander.json` |

---

## ⚙️ Física de la Simulación

- **Gravedad**: 0.18 px/frame² constante
- **Empuje máximo**: 0.55 (supera la gravedad si potencia > 0.33)
- **Inercia de rotación**: el ángulo cambia suavemente (15% por frame hacia objetivo)
- **Combustible**: 100 unidades, se consume 0.35 × potencia por frame
- **Plataforma**: se mueve horizontalmente ±0.8 px/frame, rebota en los bordes
- **Condición de aterrizaje exitoso**:
  - `|vx| ≤ 1.8` px/frame
  - `|vy| ≤ 2.5` px/frame
  - `|ángulo| ≤ 15°`
  - Centro de la nave dentro del pad

---

## 📊 Comparación con Dino GA

| Característica | Dino GA | Alien Lander GA |
|----------------|---------|-----------------|
| Cromosoma | Pesos de red neuronal | Secuencia de comandos directos |
| Genes | 81 reales | 240 reales |
| Objetivo | Distancia máxima | Aterrizaje preciso |
| Dificultad | Obstáculos aleatorios | Plataforma móvil + fuel |
| Fitness | Simple (frames) | Multi-objetivo |
| Cruce | 1 punto | BLX-α + 2 puntos |