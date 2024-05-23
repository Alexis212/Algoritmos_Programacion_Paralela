"""Script para listar los primeros N primos usando el metodo criba de eratostenes."""

from mpi4py import MPI
import sys
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def is_div(a, b):
    """Verifica si el primer numero es multiplo del segundo."""
    return a % b == 0

def eratostenes(num):
    """Implementa el algoritmo de criba de eratostenes para encontrar numeros primos."""
    thread_part = int(num / size)  # Segmento que tendra cada parte cada proceso
    remainder = num % size  # El resto o segmento sobrante

    inc = 1  # Incrementador
    share_variables = {"current_prime": 2, "total_length": num-1}  # Variables compartidas
    current_prime = 2  # El ultimo primo que hemos identificado
    total_lenght = num-1  # La longitud total del arreglo de primos

    if rank < size-1:
        # Creamos un arreglo de tamaño thread_part empezando por el numero 2...
        xs = np.arange(rank*thread_part + 2, (rank+1)*thread_part + 2)

    else:
        # ...en la ultima linea añadimos el resto
        xs = np.arange(rank*thread_part + 2, (rank+1)*thread_part + remainder)

    # Iteramos mientras el incrementador sea menor que el tamaño actual de la lista de primos
    while inc < share_variables["total_length"]:
        ys = []
        for x in xs:
            if not is_div(x, share_variables["current_prime"]) or x == share_variables["current_prime"]:
                ys.append(x)

        primes = comm.gather(ys, root=0)  # Enviamos todas las listas de primos al proceso 0
        if rank == 0:
            primes = sorted(primes, key=lambda x: x[0])  # Ordena las listas según su primer elemento...
            all_primes = np.concatenate(primes)  # ...y las concatena en una unica lista.

            # Actualizamos las variables compartidas
            share_variables["total_length"] = len(all_primes)
            share_variables["current_prime"] = all_primes[inc]

        # Volvemos a enviar las variables compartidas y volvemos a repartir la lista actualizada a los procesos
        share_variables = comm.bcast(share_variables, root=0)
        xs = comm.scatter(primes, root=0)
        inc += 1  # Actualizamos el incrementador

    # Una vez salimos del bucle, el proceso a terminado
    if rank == 0:
        primes = sorted(primes, key=lambda x: x[0])  # Ordenamos la lista según su primer elemento...
        all_primes = np.concatenate(primes)  # ...y las concatenamos en una unica lista.
        print( len(all_primes) )


if __name__ == '__main__':
    num = int(sys.argv[1]) + 1
    eratostenes(num)
