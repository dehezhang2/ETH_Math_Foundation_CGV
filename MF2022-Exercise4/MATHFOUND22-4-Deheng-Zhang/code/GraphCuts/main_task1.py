from graph_cut import GraphCut
import numpy as np


def main():
    optimizer = GraphCut(3, 4)
    optimizer.set_unary(np.array([
        [4, 9],
        [7, 7],
        [8, 5]
    ]))
    optimizer.set_pairwise(np.array([
        [0, 1, 0, 3, 2, 0],
        [1, 2, 0, 5, 1, 0]
    ]))
    print("Optimized energy: %lf"%optimizer.minimize())
    labels = optimizer.get_labeling()
    for i in range(3):
        print("label of node %d: %d"%(i, labels[i]))
    
    pass


if __name__ == '__main__':
    main()
