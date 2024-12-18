import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np


# Plot and show execution times chart
# sequential -> Map consisting of relation between number of processes and array of multiple tries
# parallel -> Map consisting of relation between number of processes and array of multiple tries
# parallel_name -> MPI or OpenMPI
def execution_chart(sequential: dict, parallel: dict, parallel_name: str, convertion_name: str):

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
    for x, y in zip(seq_keys, seq_means):
        plt.text(x, y+0.05, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')
    plt.plot(
        par_keys,
        par_means,
        marker='o',
        label=f"Paralelismo con {parallel_name}",
        color="blue"
    )
    for x, y in zip(par_keys, par_means):
        plt.text(x, y+0.05, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')
    plt.xlabel("Número de procesos")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title(f"Tiempos de ejecución, {convertion_name}: Secuencial vs {parallel_name}")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.3f}')  # Four decimals on axis Y chart
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
    plt.show()

# Plot and show execution times chart [ALLOWING MULTIPLE PARALLELISMS for OMP or MPI]
def execution_chart_v3(sequential: dict, parallel: dict, parallel_name: str, convertion_name: str):
    # Process sequential data
    seq_keys = list(sequential.keys())
    seq_means = [np.mean(values) for values in sequential.values()]

    # Plot sequential data
    plt.figure(figsize=(10, 5))
    plt.plot(
        seq_keys,
        seq_means,
        marker='o',
        linestyle='--',
        label="Secuencial",
        color="red"
    )
    for x, y in zip(seq_keys, seq_means):
        plt.text(x, y, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')

    # Process and plot parallel data (handling multiple series)
    for primary_key, secondary_data in parallel.items():
        par_keys = list(secondary_data.keys())
        par_means = [np.mean(values) for values in secondary_data.values()]

        # Color determinista basado en el índice o primary_key
        color_not_random = get_deterministic_color_execution(primary_key)

        plt.plot(
            par_keys,
            par_means,
            marker='o',
            label=f"Paralelismo {parallel_name} - {primary_key} Nodos",
            linestyle='-',  # Solid line for parallelism
            color=color_not_random 
        )
        for x, y in zip(par_keys, par_means):
            plt.text(x, y + 0.025, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')

    # Final touches for the chart
    plt.xlabel("Número de procesos")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title(f"Tiempos de ejecución, {convertion_name}: Secuencial vs {parallel_name}")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.3f}')  # Four decimals on axis Y chart
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
    plt.show()

# Plot and show execution times chart based on threads [ALLOWING MULTIPLE PARALLELISMS for OMP+MPI]
def execution_chart_v4(sequential: dict, parallel: dict, parallel_name: str, convertion_name: str):
    plt.figure(figsize=(10, 5))

    # Process and plot sequential data
    seq_keys = list(sequential.keys())
    seq_means = [np.mean(values) for values in sequential.values()]

    plt.plot(
        seq_keys,
        seq_means,
        marker='o',
        linestyle='--',
        label="Secuencial",
        color="red"
    )
    for x, y in zip(seq_keys, seq_means):
        plt.text(x, y, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')

    # Process and plot parallel data (handling multiple keys: primary, secondary, tertiary)
    for primary_key, secondary_data in parallel.items():
        for secondary_key, tertiary_data in secondary_data.items():
            # Process the third level (tertiary key) and extract means
            par_keys = list(tertiary_data.keys())
            par_means = [np.mean(tertiary_data[tertiary_key]) for tertiary_key in par_keys]

            # Deterministic color based on the primary_key to differentiate series
            color_not_random = get_deterministic_color_execution(primary_key)

            plt.plot(
                par_keys,
                par_means,
                marker='o',
                label=f"Paralelismo {parallel_name} - {primary_key} Nodos, {secondary_key} Procesos",
                linestyle='-',  # Solid line for parallelism
                color=color_not_random
            )

            # Annotate the parallel data points
            for x, y in zip(par_keys, par_means):
                plt.text(x, y + 0.025, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')

    # Final touches for the chart
    plt.xlabel("Número de hilos")
    plt.ylabel("Tiempo de ejecución (s)")
    plt.title(f"Tiempos de ejecución, {convertion_name}: Secuencial vs {parallel_name}")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.3f}')  # Four decimals on axis Y chart
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
    plt.show()

def get_deterministic_color_execution(index: int):
    # Lista de colores predeterminada para asegurar que los colores sean siempre los mismos
    color_list = [
        'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta'
    ]
    return color_list[index-1 % (len(color_list))]  # Cicla a través de la lista si hay más claves que colores


# Plot and show speed-up metrics chart [OMP]
# sequential -> Map consisting of relation between number of processes and array of multiple tries
# parallel -> Map consisting of relation between number of processes and array of multiple tries
# parallel_name -> MPI or OpenMPI
# convertion_name -> HSL or YUV
def speed_up_chart(sequential: dict, parallel: dict, parallel_name: str, convertion_name: str):

    processes = list(sequential.keys())  # sequential or parallel is valid

    seq_means = [np.mean(values) for values in sequential.values()]
    par_means = [np.mean(values) for values in parallel.values()]
    speed_up = [seq / par for seq, par in list(zip(seq_means, par_means))]

    plt.figure(figsize=(10, 5))
    for x, y in zip(processes, speed_up):
        plt.text(x, y+0.05, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')
    plt.plot(processes, speed_up, marker='o', color='orange', label="Speed-Up")
    plt.xlabel("Número de procesos")
    plt.ylabel("Aceleración")
    plt.title(f"{parallel_name}, {convertion_name}: Aceleración en relación al número de procesos")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.3f}')  # Four decimals on axis Y chart
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
    plt.show()

# Plot and show speed-up metrics chart [ALLOWING MULTIPLE PARALLELISMS for OMP or MPI]
def speed_up_chart_v3(sequential: dict, parallel: dict, parallel_name: str, convertion_name: str):
    plt.figure(figsize=(10, 5))

    # Process sequential data
    seq_keys = list(sequential.keys())
    seq_means = {key: np.mean(values) for key, values in sequential.items()}

    # Iterate through parallel data (primary and secondary keys)
    for primary_key, secondary_data in parallel.items():
        # Find the common keys between sequential and this parallel group
        common_keys = list(set(seq_keys) & set(secondary_data.keys()))
        common_keys.sort()

        if not common_keys:
            continue  # Skip if no common keys for this primary_key group

        # Calculate the mean values for the common keys
        par_means = [np.mean(secondary_data[key]) for key in common_keys]
        seq_means_for_common_keys = [seq_means[key] for key in common_keys]

        # Calculate speed-up (sequential time / parallel time)
        speed_up = [seq / par for seq, par in zip(seq_means_for_common_keys, par_means)]

        # Color determinista basado en el primary_key
        color_speedup = get_deterministic_color_speedup(primary_key)

        # Plot the speed-up chart for each parallel group
        plt.plot(
            common_keys,
            speed_up,
            marker='o',
            label=f"Aceleración {parallel_name} - {primary_key} Nodos",
            color=color_speedup
        )

        # Annotate each point with its speed-up value
        for x, y in zip(common_keys, speed_up):
            plt.text(x, y + 0.05, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')

    # Final touches for the chart
    plt.xlabel("Número de procesos")
    plt.ylabel("Aceleración")
    plt.title(f"{parallel_name}, {convertion_name}: Aceleración en relación al número de procesos")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.3f}')  # Four decimals on axis Y chart
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
    plt.show()

# Plot and show speed-up metrics chart [ALLOWING MULTIPLE PARALLELISMS for OMP+MPI]
def speed_up_chart_v4(sequential: dict, parallel: dict, parallel_name: str, convertion_name: str):
    plt.figure(figsize=(10, 5))

    # Process sequential data
    seq_keys = list(sequential.keys())
    seq_means = {key: np.mean(values) for key, values in sequential.items()}

    # Iterate through parallel data (primary and secondary keys)
    for primary_key, secondary_data in parallel.items():
        # Iterate through each process (secondary_key) and each thread (tertiary_key)
        for secondary_key, tertiary_data in secondary_data.items():
            # Find the common keys between sequential and this parallel group
            common_keys = list(set(seq_keys) & set(tertiary_data.keys()))
            common_keys.sort()

            if not common_keys:
                continue  # Skip if no common keys for this secondary_key group

            # Calculate the mean values for the common keys
            par_means = [np.mean(tertiary_data[key]) for key in common_keys]
            seq_means_for_common_keys = [seq_means[key] for key in common_keys]

            # Calculate speed-up (sequential time / parallel time) for each thread
            speed_up = [seq / par for seq, par in zip(seq_means_for_common_keys, par_means)]

            # Color determinista basado en el primary_key to differentiate series
            color_speedup = get_deterministic_color_speedup(primary_key)

            # Plot the speed-up chart for each thread in each process
            plt.plot(
                common_keys,
                speed_up,
                marker='o',
                label=f"Aceleración {parallel_name} - {primary_key} Nodos, {secondary_key} Procesos",
                color=color_speedup
            )

            # Annotate each point with its speed-up value
            for x, y in zip(common_keys, speed_up):
                plt.text(x, y + 0.05, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')

    # Final touches for the chart
    plt.xlabel("Número de hilos")
    plt.ylabel("Aceleración")
    plt.title(f"{parallel_name}, {convertion_name}: Aceleración en relación al número de hilos")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.3f}')  # Four decimals on axis Y chart
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
    plt.show()

    plt.figure(figsize=(10, 5))

    # Process sequential data
    seq_keys = list(sequential.keys())
    seq_means = {key: np.mean(values) for key, values in sequential.items()}

    # Iterate through parallel data (primary and secondary keys)
    for primary_key, secondary_data in parallel.items():
        # Iterate through each thread (tertiary_key) and calculate speed-up for each process
        for tertiary_key, process_data in secondary_data.items():
            # Find the common keys between sequential and this parallel group (processes)
            common_keys = list(set(seq_keys) & set(process_data.keys()))
            common_keys.sort()

            if not common_keys:
                continue  # Skip if no common keys for this process group

            # Calculate the mean values for the common keys (processes)
            par_means = [np.mean(process_data[key]) for key in common_keys]
            seq_means_for_common_keys = [seq_means[key] for key in common_keys]

            # Calculate speed-up (sequential time / parallel time) for each process
            speed_up = [seq / par for seq, par in zip(seq_means_for_common_keys, par_means)]

            # Color determinista basado en el tertiary_key (threads) to differentiate series
            color_speedup = get_deterministic_color_speedup(tertiary_key)

            # Plot the speed-up chart for each thread in each process
            plt.plot(
                common_keys,
                speed_up,
                marker='o',
                label=f"Aceleración {parallel_name} - {primary_key} Nodos, {tertiary_key} Hilos",
                color=color_speedup
            )

            # Annotate each point with its speed-up value
            for x, y in zip(common_keys, speed_up):
                plt.text(x, y + 0.05, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')

    # Final touches for the chart
    plt.xlabel("Número de procesos")
    plt.ylabel("Aceleración")
    plt.title(f"{parallel_name}, {convertion_name}: Aceleración en relación al número de procesos")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.3f}')  # Four decimals on axis Y chart
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
    plt.show()

def get_deterministic_color_speedup(index: int):
    # Lista de colores predeterminada para asegurar que los colores sean siempre los mismos
    color_list = [
        'orange', 'yellow', 'yellow', 'olive', 'olive', 'olive', 'olive', 'purple', 'cyan', 'magenta'
    ]
    return color_list[index-1 % (len(color_list))]  # Cicla a través de la lista si hay más claves que colores


# Plot and show real speed-up versus gustafson metrics chart
# sequential -> Map consisting of relation between number of processes and array of multiple tries
# parallel -> Map consisting of relation between number of processes and array of multiple tries
# parallel_name -> MPI or OpenMPI
# convertion_name -> HSL or YUV
def gustafson_chart(sequential: dict, parallel: dict, parallel_name: str, convertion_name: str):

    processes = list(sequential.keys())  # sequential or parallel is valid

    seq_means = [np.mean(values) for values in sequential.values()]
    par_means = [np.mean(values) for values in parallel.values()]
    speed_up = [seq / par for seq, par in list(zip(seq_means, par_means))]
    alphas = [(n_process - sp) / (n_process - 1) for n_process, sp in zip(processes[1:], speed_up[1:])]
    alpha_avg = np.mean(alphas)
    gustafson = [n_process - alpha_avg * (n_process - 1) for n_process in processes]

    plt.figure(figsize=(10, 5))
    for x, y in zip(processes, speed_up):
        plt.text(x, y, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')
    plt.plot(processes, speed_up, marker='o', color='orange', label="Speed-Up empírico")
    plt.plot(processes, gustafson, marker='o', linestyle='--', color='blue', label="Gustafson")
    plt.xlabel("Número de procesos")
    plt.ylabel("Aceleración")
    plt.title(f"{parallel_name}, {convertion_name}: Aceleración teórica (Gustafson) vs aceleración empírica")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.3f}')  # Four decimals on axis Y chart
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.50))
    plt.show()


# Function to calculate Gustafson's Law
def gustafson_law(processes, seq_means, par_means):
    # Calculate speed-up for each number of processes
    speed_up = [seq / par for seq, par in zip(seq_means, par_means)]
    alphas = [(n_process - sp) / (n_process - 1) for n_process, sp in zip(processes[1:], speed_up[1:])]
    alpha_avg = np.mean(alphas)
    
    # Gustafson's Law: S(N) = N - alpha_avg * (N - 1)
    gustafson = [n_process - alpha_avg * (n_process - 1) for n_process in processes]
    
    return gustafson, speed_up

# Function to plot speed-up chart with Gustafson's Law [FOR OMP or MPI]
def speed_up_chart_v3_with_gustafson(sequential: dict, parallel: dict, parallel_name: str, convertion_name: str):
    plt.figure(figsize=(10, 5))

    # Process sequential data
    seq_keys = list(sequential.keys())
    seq_means = {key: np.mean(values) for key, values in sequential.items()}

    # Iterate through parallel data (primary and secondary keys)
    for primary_key, secondary_data in parallel.items():
        # Find the common keys between sequential and this parallel group
        common_keys = list(set(seq_keys) & set(secondary_data.keys()))
        common_keys.sort()

        if not common_keys:
            continue  # Skip if no common keys for this primary_key group

        # Calculate the mean values for the common keys
        par_means = [np.mean(secondary_data[key]) for key in common_keys]
        seq_means_for_common_keys = [seq_means[key] for key in common_keys]

        # Calculate empirical speed-up (sequential time / parallel time)
        speed_up = [seq / par for seq, par in zip(seq_means_for_common_keys, par_means)]

        # Calculate Gustafson's Law
        gustafson, _ = gustafson_law(common_keys, seq_means_for_common_keys, par_means)

        # Color based on the primary_key
        color_speedup = get_deterministic_color_speedup(primary_key)

        # Plot the empirical speed-up for each parallel group
        plt.plot(
            common_keys,
            speed_up,
            marker='o',
            label=f"Aceleración {parallel_name} - {primary_key} Nodos (Empírico)",
            color=color_speedup
        )

        # Plot the Gustafson curve
        plt.plot(
            common_keys,
            gustafson,
            marker='',
            linestyle='--',
            label=f"Gustafson - {primary_key} Nodos",
            color='blue'
        )

        # Annotate each point with its speed-up value
        for x, y in zip(common_keys, speed_up):
            plt.text(x, y + 0.05, f'{y:.3f}', fontsize=10, ha='center', va='bottom', color='black')

    # Final touches for the chart
    plt.xlabel("Número de procesos")
    plt.ylabel("Aceleración")
    plt.title(f"{parallel_name}, {convertion_name}: Aceleración empírica vs Aceleración teórica (Gustafson)")
    plt.grid(True)
    plt.legend()
    formatter = FuncFormatter(lambda x, _: f'{x:.3f}')  # Four decimals on axis Y chart
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.25))
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

# Process user's input [ALLOWING MULTIPLE PARALLELISM for OMP or MPI]
# Accepted format: S 1 0.5 0.4 0.6 where S indicates sequential, first number the key and the others the values
def process_input_v3():
    sequential = {}
    parallel = {}

    print("Introduce los datos en el formato especificado (e.g., 'S 1 0.5 0.4 0.6' o 'P 2 1 0.3 0.2 0.1').")
    print("Introduce 'FINISH' para finalizar la entrada.")

    while True:
        user_input = input().strip()
        if user_input == "FINISH":
            break
        split_input = user_input.split()
        if len(split_input) < 4 and split_input[0] == "P":
            print("Error: Formato inválido. Para paralelo, debes introducir al menos una clave primaria, una clave secundaria y tres valores.")
            continue
        if len(split_input) < 3 and split_input[0] == "S":
            print("Error: Formato inválido. Para secuencial, debes introducir al menos una clave y tres valores.")
            continue

        program_type = split_input[0]

        if program_type == "S":
            key, *values = split_input[1:]
            key = int(key)
            values = list(map(float, values))
            if key in sequential:
                print(f"Advertencia: Sobrescribiendo clave {key} en secuencial.")
            sequential[key] = values

        elif program_type == "P":
            primary_key, secondary_key, *values = split_input[1:]
            primary_key = int(primary_key)
            secondary_key = int(secondary_key)
            values = list(map(float, values))

            if primary_key not in parallel:
                parallel[primary_key] = {}

            if secondary_key in parallel[primary_key]:
                print(f"Advertencia: Sobrescribiendo clave ({primary_key}, {secondary_key}) en paralelo.")
            
            parallel[primary_key][secondary_key] = values

        else:
            print("Error: Tipo no reconocido. Usa 'S' para secuencial o 'P' para paralelo.")

    return sequential, parallel

# Process user's input [ALLOWING MULTIPLE PARALLELISM for OMP+MPI]
# Accepted format: S 1 0.5 0.4 0.6 where S indicates sequential, first number the key and the others the values
def process_input_v4():
    sequential = {}
    parallel = {}

    print("Introduce los datos en el formato especificado (e.g., 'S 1 0.5 0.4 0.6' o 'P 2 1 3 0.3 0.2 0.1').")
    print("Introduce 'FINISH' para finalizar la entrada.")

    while True:
        user_input = input().strip()
        if user_input == "FINISH":
            break
        split_input = user_input.split()

        # Handle sequential input validation
        if len(split_input) < 3 and split_input[0] == "S":
            print("Error: Formato inválido. Para secuencial, debes introducir al menos una clave y tres valores.")
            continue

        # Handle parallel input validation (now requires three keys)
        if len(split_input) < 5 and split_input[0] == "P":
            print("Error: Formato inválido. Para paralelo, debes introducir al menos una clave primaria, una clave secundaria, una clave terciaria y tres valores.")
            continue

        program_type = split_input[0]

        # Sequential data processing
        if program_type == "S":
            key, *values = split_input[1:]
            key = int(key)
            values = list(map(float, values))
            if key in sequential:
                print(f"Advertencia: Sobrescribiendo clave {key} en secuencial.")
            sequential[key] = values

        # Parallel data processing with three keys (primary, secondary, tertiary)
        elif program_type == "P":
            primary_key, secondary_key, tertiary_key, *values = split_input[1:]
            primary_key = int(primary_key)
            secondary_key = int(secondary_key)
            tertiary_key = int(tertiary_key)
            values = list(map(float, values))

            # Ensure the primary and secondary keys exist in the nested dictionary
            if primary_key not in parallel:
                parallel[primary_key] = {}
            if secondary_key not in parallel[primary_key]:
                parallel[primary_key][secondary_key] = {}

            # Insert or update the parallel data
            if tertiary_key in parallel[primary_key][secondary_key]:
                print(f"Advertencia: Sobrescribiendo clave ({primary_key}, {secondary_key}, {tertiary_key}) en paralelo.")

            parallel[primary_key][secondary_key][tertiary_key] = values

        else:
            print("Error: Tipo no reconocido. Usa 'S' para secuencial o 'P' para paralelo.")

    return sequential, parallel


# Validate user's input
# At least all keys should be present and three attempts should be done
# 1, 2, 4, 8 and 16 processes with more than 3 tries per process
def validate_user_input(sequential: dict, parallel: dict):

    required_keys = {1, 2, 4, 8, 16}
    minimum_attempts = 1

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

# Validate user's input [RELAXED CONSTRAINTS TO ALLOW LESS KEYS/ATTEMPTS]
# At least one key should be present and the corresponding process should have enough attempts
def validate_user_input_v2(sequential: dict, parallel: dict):

    minimum_attempts = 1

    assert len(sequential) > 0, "El diccionario secuencial no puede estar vacío."
    for clave, valores in sequential.items():
        assert len(valores) >= minimum_attempts, \
            (f"La clave {clave} en secuencial debe tener al menos {minimum_attempts} intentos. "
             f"Valores actuales: {valores}")

    assert len(parallel) > 0, "El diccionario paralelo no puede estar vacío."
    for clave, valores in parallel.items():
        assert len(valores) >= minimum_attempts, \
            f"La clave {clave} en paralelo debe tener al menos {minimum_attempts} intentos. Valores actuales: {valores}"



# Request parallel name for user's case [OMP]
# Only accepted values -> "OpenMP", "MPI" or "OpenMPI y MPI"
def request_parallel_name():
    while True:
        paralelismo = input("Introduce el paralelismo utilizado (OpenMP, MPI, OpenMP y MPI): ").strip()
        if paralelismo in {"OpenMP", "MPI", "OpenMP y MPI"}:
            return paralelismo
        else:
            print(f"Paralelismo inválido. Debe ser 'OpenMP', 'MPI' o 'OpenMP y MPI'. Valor recibido: {paralelismo}")

# Request parallel name for user's case [ALLOWING RELAXED TITLES (MPI 1 core as opposed to only MPI)]
# Only accepted values -> "OpenMP", "MPI" or "OpenMPI y MPI"
def request_parallel_name_v2():
    while True:
        paralelismo = input("Introduce el paralelismo utilizado (OpenMP, MPI, OpenMP y MPI): ").strip()
        return paralelismo


# Request convertion name for user's case
# Only accepted values -> "HSL" or "YUV"
def request_convertion_name():
    while True:
        convertion = input("Introduce la conversión medida (grey, HSL o YUV): ").strip()
        if convertion in {"grey", "HSL", "YUV"}:
            return convertion
        else:
            print(f"Conversión inválida. Debe ser 'grey', 'HSL' o 'YUV'. Valor recibido: {convertion}")


def main():
    sequential, parallel = process_input_v3()

    try:
        validate_user_input_v2(sequential, parallel) # Relaxed to allow more diverse inputs
    except AssertionError as assertion_error:
        print(f"Error de validación: {assertion_error}")
        return

    parallel_name = request_parallel_name_v2() # Relaxed to allow more diverse inputs
    convertion_name = request_convertion_name()

    print("Los datos de entrada son válidos, generando gráficas...")
    execution_chart_v3(sequential, parallel, parallel_name, convertion_name) # Relaxed to allow more diverse inputs
    speed_up_chart_v3(sequential, parallel, parallel_name, convertion_name) # Relaxed to allow more diverse inputs
    speed_up_chart_v3_with_gustafson(sequential, parallel, parallel_name, convertion_name)

if __name__ == "__main__":
    main()
