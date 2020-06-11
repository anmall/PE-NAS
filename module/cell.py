import numpy as np
import copy
import itertools
import random
from nasbench import api


INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


class Cell:

    def __init__(self, matrix, ops):

        self.matrix = matrix
        self.ops = ops

    def serialize(self):
        return {
            'matrix': self.matrix.tolist(),
            'ops': self.ops
        }

    def modelspec(self):
        return api.ModelSpec(matrix=self.matrix, ops=self.ops)

    @classmethod
    def random_cell(cls, nasbench):
        """ 
        From the NASBench repository 
        https://github.com/google-research/nasbench
        """
        while True:
            matrix = np.random.choice(
                [0, 1], size=(NUM_VERTICES, NUM_VERTICES))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(OPS, size=NUM_VERTICES).tolist()
            ops[0] = INPUT
            ops[-1] = OUTPUT
            spec = api.ModelSpec(matrix=matrix, ops=ops)
            if nasbench.is_valid(spec):

                return Cell(matrix, ops)



    def mutate(self, nasbench, mutation_rate=1.0):
        """
        similar to perturb. A stochastic approach to perturbing the cell
        inspird by https://github.com/google-research/nasbench
        """
        while True:
            new_matrix = copy.deepcopy(self.matrix)
            new_ops = copy.deepcopy(self.ops)

            edge_mutation_prob = mutation_rate / NUM_VERTICES
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = mutation_rate / OP_SPOTS
            for ind in range(1, OP_SPOTS + 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in OPS if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)

            new_spec = api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return Cell(new_matrix, new_ops)

    def encode_cell(self):
        """ 
        compute the "standard" encoding,
        i.e. adjacency matrix + op list encoding 
        """
        encoding_length = (NUM_VERTICES ** 2 - NUM_VERTICES) // 2 + OP_SPOTS
        encoding = np.zeros((encoding_length))
        dic = {CONV1X1: 0., CONV3X3: 0.5, MAXPOOL3X3: 1.0}
        n = 0
        for i in range(NUM_VERTICES - 1):
            for j in range(i+1, NUM_VERTICES):
                encoding[n] = self.matrix[i][j]
                n += 1
        for i in range(1, NUM_VERTICES - 1):
            encoding[-i] = dic[self.ops[i]]
        return encoding.tolist()



