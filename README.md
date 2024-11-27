# Práctica 2: Paralelización de código con MPI + OpenMP
By Luis Daniel Casais Mezquida, Lucas Gallego Bravo, Francisco Montañés de Lucas & Diego Picazo García  
Computación de Altas Prestaciones 24/25  
Máster en Ingeniería Informática  
Universidad Carlos III de Madrid


## Enunciado de la práctica
La mejora de contraste es una operación común en el procesamiento de imagen. Es un método útil para el procesamiento de imágenes científicas, tales como imágenes de rayos X o imágenes obtenidas por satélite. También es una técnica útil para mejorar los detalles en las fotografías que están sobre o sub expuestas.

El objetivo de este trabajo es desarrollar una aplicación de mejora de contraste que utiliza la aceleración sobre [MPICH](https://www.mpich.org/) y [OpenMP](https://www.openmp.org/).

En este documento, se dan las ideas básicas y los principios de mejora de contraste mediante modificación del histograma. Empezamos con el caso simple, realce de contraste para las imágenes en escala de grises. Entonces, tratamos de aplicar el método similar a las imágenes en color.

Una implementación sencilla en C se proporciona como referencia y punto de partida.

El histograma de una imagen digital representa su distribución tonal. La ecualización del histograma indica los valores de intensidad más frecuentes. Esto permite conocer las áreas de menor contraste para obtener un mayor contraste sin afectar el contraste global de la imagen. La siguiente imagen muestra el efecto de la ecualización del histograma en el histograma de la imagen.

La ecualización de histograma se puede realizar en los siguientes pasos:
1. Calcular el histograma de la imagen de entrada
2. Calcular la distribución acumulativa del histograma
3. Utilización de la distribución acumulativa para construir una tabla de búsqueda que mapea cada valor de gris a la ecualizada (este paso se puede combinar con la última)
4. Actualización de la imagen utilizando la tabla de búsqueda construida en el último paso


### Requisitos de programación
Se debe implementar un programa paralelo en MPI y OpenMP que realice eficientemente los cálculos. Se proporciona una versión secuencial del programa.

Algunas observaciones sobre la ejecución de la aplicación:
- La aplicación utiliza archivos PGM (para imágenes en escala de grises) y PPM (para imágenes en color). Por defecto, el nombre de archivo del archivo de entrada es `in.pgm` y `in.ppm` respectivamente.
- Dado que el tamaño de las imágenes PPM/PGM es demasiado grande, se dispone de la siguiente imagen en formato JPG. Una vez descargada, será necesario convertirla al formato PPM/PGM (se puede utilizar el software [`convert`](https://www.imagemagick.org/script/convert.php)).
- El ejecutable requiere de los archivos de entrada `in.pgm` y `in.ppm` y genera los archivos procesados
`out_hsl.ppm` y `out_yuv.ppm`. No es necesario indicar parámetros de entrada.

El programa se ejecutará en el servidor `avignon.lab.inf.uc3m.es` con un sistema de colas [SLURM](https://slurm.schedmd.com/) de la siguiente manera:
```
srun -p gpus -N 1 -n 1 ./contrast
```
Donde el parámetro `-N` corresponde el número de computadores y `-n` con el número de núcleos por computador. Se recomienda el uso de los computadores de la cola `gpus` del laboratorio para los experimentos. Estos cuatro computadores cuentan con 12 núcleos cada uno, 48 en total en forma distribuida.

La versión paralela del programa consistirá de los siguientes pasos:
- Los procesos deben computar una sección reducida para reducir la carga de cómputo de cada proceso.
- Únicamente un proceso (_rank_ `0` por ejemplo) leerá las imágenes y escribirá los resultados en el disco. Además, un único proceso presentará el tiempo total de la aplicación.

### Pistas
- Usa el comando `convert` para poder generar y visualizar las imágenes procesadas.
  Por ejemplo:
  ```
  convert highres.jpg in.pgm
  ```
- Para tomar tiempos se recomienda usar la función `MPI_Wtime()` de la siguiente forma:
  ```cpp
  double tstart = MPI_Wtime();

  // código

  double tfinish = MPI_Wtime();
  double TotalTime = tfinish - tstart;
  ```
- Para asegurarse del correcto funcionamiento de la versión paralela, en comparación con la versión
secuencial, seguid los siguientes pasos:
  1. Ejecutar la versión secuencial con un conjunto de datos de entrada.
  2. Ejecutar las diferentes versiones paralelas con el mismo conjunto de datos de entrada que la versión secuencial.
  3. Comparar los conjuntos de datos de salida de cada versión paralela con el obtenido en la versión secuencial usando el comando `diff` (`man diff`).
  4. Si el comando `diff` detecta diferencias entre el fichero de salida de la versión secuencial y alguno de los ficheros de salida de las versiones paralelas se considerará que el programa no funciona correctamente y, por tanto, habrá que revisar el algoritmo paralelo.


## Compilación
Es necesario tener instalado [OpenMP](https://www.openmp.org/) (suele venir ya instalado con el compilador GCC) y [MPICH](https://www.mpich.org/).

Compila con:
```
mkdir build
cd build/
cmake .. -DCMAKE_CXX_COMPILER=mpicxx.mpich
make
```

Esto generará cuatro ejecutables (en `build/`): `contrast`, `contrast-mpi`, `contrast-omp` y `contrast-mpi-omp`, las cuales corresponden con las cuatro implementaciones.


## Ejecución
Es necesario


Para ejecutar:
```
srun -p gpus -N <nodos> -n <cores> <ejecutable>
```




> [!NOTE]
> El máximo de nodos en `avignon` es de 4, y un máximo de 12 cores por nodo.

> [!TIP]
> También puedes ejecutarlo en batch, para lo que es necesario un archivo `run.sh`:
> ```
> #!/bin/bash
>
> #SBATCH --job-name=contrast-<version>
> #SBATCH --output=contrast-<version>_%j.log
> #SBATCH --time=00:00:30  # max duration
> #SBATCH --ntasks=<cores>
> #SBATCH --nodes=<nodos>
> #SBATCH --cpus-per-task=1  # cpus per task/cores
> #SBATCH --partition=gpus
> 
> srun --mpi=pmix <ejecutable>
> ```
>
> Y después se ejecuta con:
> ```
> sbatch run.sh
> ```
