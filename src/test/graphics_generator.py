import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


# Plot and show execution times graphic
# sequential -> Map consisting of relation between number of processes and array of multiple tries
# parallel -> Map consisting of relation between number of processes and array of multiple tries
# parallel_name -> MPI or OpenMPI
def execution_graphic(sequential: dict, parallel: dict, parallel_name: str):

    seq_keys = list(sequential.keys())
    seq_means = [np.mean(values) for values in sequential.values()]

    par_keys = list(parallel.keys())
    par_means = [np.mean(values) for values in parallel.values()]

    plt.figure(figsize=(10, 5))
    plt.plot(
        seq_keys,
        seq_means,
        marker='o',
        linestyle='--',
        label="Secuencial",
        color="red"
    )
    plt.plot(
        par_keys,
        par_means,
        marker='o',
        label=f"Paralelismo con {parallel_name}",
        color="blue"
    )
    plt.xlabel("Número de procesos")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title(f"Tiempos de ejecución: Secuencial vs {parallel_name}")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.4f}')  # Four decimals on graphic
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.show()


# Plot and show speed-up metrics graphic
# sequential -> Map consisting of relation between number of processes and array of multiple tries
# parallel -> Map consisting of relation between number of processes and array of multiple tries
# parallel_name -> MPI or OpenMPI
def speed_up_graphic(sequential: dict, parallel: dict, parallel_name: str):

    processes = list(sequential.keys())  # sequential or parallel is valid

    seq_means = [np.mean(values) for values in sequential.values()]
    par_means = [np.mean(values) for values in parallel.values()]
    speed_up = [seq / par for seq, par in list(zip(seq_means, par_means))]

    plt.figure(figsize=(10, 5))
    plt.plot(processes, speed_up, marker='o', color='orange', label="Speed-Up")
    plt.xlabel("Número de procesos")
    plt.ylabel("Aceleración")
    plt.title(f"{parallel_name}: Aceleración en relación al número de procesos")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.4f}')  # Four decimals on graphic
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.show()


# Process user's input
# Accepted format: S 1 0.5 0.4 0.6 where S indicates sequential, first number the key and the others the values
def process_input():

    sequential = {}
    parallel = {}

    print("Introduce los datos en el formato especificado (e.g., 'S 1 0.5 0.4 0.6' o 'P 2 0.3 0.2 0.1').")
    print("Introduce 'FINISH' para finalizar la entrada.")

    while True:
        user_input = input().strip()
        if user_input == "FINISH":
            break
        split_input = user_input.split()
        if len(split_input) < 3:
            print("Error: Formato inválido. Debes introducir al menos una clave y tres valores.")
            continue
        program_type, key, *values = split_input
        key = int(key)
        values = list(map(float, values))
        if program_type == "S":
            if key in sequential:
                print(f"Advertencia: Sobrescribiendo clave {key} en secuencial.")
            sequential[key] = values
        elif program_type == "P":
            if key in parallel:
                print(f"Advertencia: Sobrescribiendo clave {key} en paralelo.")
            parallel[key] = values
        else:
            print("Error: Tipo no reconocido. Usa 'S' para secuencial o 'P' para paralelo.")

    return sequential, parallel


# Validate user's input
# At least all keys should be present and three attempts should be done
# 1, 2, 4, 8 and 16 processes with more than 3 tries per process
def validate_user_input(sequential: dict, parallel: dict):

    required_keys = {1, 2, 4, 8, 16}
    minimum_attempts = 3

    assert set(sequential.keys()) == required_keys, \
        f"El diccionario secuencial debe contener las claves {required_keys}. Claves actuales: {sequential.keys()}"
    for clave, valores in sequential.items():
        assert len(valores) >= minimum_attempts, \
            (f"La clave {clave} en secuencial debe tener al menos {minimum_attempts} intentos. "
             f"Valores actuales: {valores}")

    assert set(parallel.keys()) == required_keys, \
        f"El diccionario paralelo debe contener las claves {required_keys}. Claves actuales: {parallel.keys()}"
    for clave, valores in parallel.items():
        assert len(valores) >= minimum_attempts, \
            f"La clave {clave} en paralelo debe tener al menos {minimum_attempts} intentos. Valores actuales: {valores}"


# Request parallel name for user's case
# Only accepted values -> "OpenMP", "MPI" or "OpenMPI y MPI"
def request_parallel_name():

    while True:
        paralelismo = input("Introduce el paralelismo utilizado (OpenMP, MPI, OpenMP y MPI): ").strip()
        if paralelismo in {"OpenMP", "MPI", "OpenMP y MPI"}:
            return paralelismo
        else:
            print(f"Paralelismo inválido. Debe ser 'OpenMP', 'MPI' o 'OpenMP y MPI'. Valor recibido: {paralelismo}")


def main():

    sequential, parallel = process_input()

    try:
        validate_user_input(sequential, parallel)
    except AssertionError as assertion_error:
        print(f"Error de validación: {assertion_error}")
        return

    parallel_name = request_parallel_name()

    print("Los datos de entrada son válidos, generando gráficas...")
    execution_graphic(sequential, parallel, parallel_name)
    speed_up_graphic(sequential, parallel, parallel_name)


if __name__ == "__main__":
    main()
