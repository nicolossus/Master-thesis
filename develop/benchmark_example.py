from benchmark_tools import timer


@timer
def waste_some_time(num_times):
    for _ in range(num_times):
        sum([i**2 for i in range(10000)])


if __name__ == "__main__":
    # waste_some_time(100)
    #N = 5
    #P = 3
    #T = (N**2 - 1) // P + P
    # print(T)

    print(2 + 2)

    def cT(P, N=5):
        return (N**2 - 1) // P + P

    P = [i for i in range(1, 26)]
    print(P)

    for i in range(len(P)):
        T = cT(P[i])
        print(f"Workers: {P[i]} | Time: {T}")
