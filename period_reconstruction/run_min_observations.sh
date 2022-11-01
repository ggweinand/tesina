#!/bin/bash

### Configuración del trabajo

### Nombre de la tarea
#SBATCH --job-name=min_observations

### Cola a usar (gpu, mono, multi)
#SBATCH --partition=short

### Cantidad de nodos a usar
### mono/gpu: 1
### multi:    2-8
#SBATCH --nodes=1

### Cores a utilizar por nodo = procesos por nodo * cores por proceso
### mono/gpu: <= 8
### multi:    16-20
### Cantidad de procesos a lanzar por nodo
#SBATCH --ntasks-per-node=1
### Cores por proceso (para MPI+OpenMP)
#SBATCH --cpus-per-task=1

### GPUs por nodo
### mono/multi:        0
### gpu (Tesla K20X):  1-2
### gpu (Tesla M2090): 1-3
#SBATCH --gres=gpu:0

### Tiempo de ejecucion. Formato dias-horas:minutos.
### mono/gpu: <= 7 días
### multi:    <= 4 días
#SBATCH --time 0-0:03

#---------------------------------------------------

# Script que se ejecuta al arrancar el trabajo

# Cargar el entorno del usuario incluyendo la funcionalidad de modules
# No tocar
. /etc/profile

# Configurar OpenMP y otras bibliotecas que usan threads
# usando los valores especificados arriba

# Cargar los módulos para la tarea

# Lanzar el programa
srun python min_observations.py b278
