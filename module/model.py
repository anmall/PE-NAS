import time
import numpy as np
import random
from module.cell import Cell
import copy
from copy import deepcopy



class Model(object):

    def __init__(self):
        self.arch = None
        self.accuracy = None
        self.es_accuracy = None
        self.test_accuracy = None
        self.cost = 0
        self.training_time = 0
        self.params = None
        self.time = time.time()
        self.parent = None
        self.root = None
        self.finished_time = None
        self.budget = None
        self.es = False

    def serialize(self):
        return {
            "acc": self.accuracy,
            "training_time": self.training_time,
            "params": self.params,
            "arch": self.arch.serialize(),
            "cost": self.cost,
            'test': self.test_accuracy
        }

    def __str__(self):
        return '{0:b}'.format(self.arch)

    def __lt__(self, other):
        return self.time < other.time


    def fitness(self):
        return self.accuracy


class ModelHelper:



    @classmethod
    def random_cell(cls, profiler):

        while True:
            matrix = np.random.choice(
                [0, 1], size=(7, 7))
            matrix = np.triu(matrix, 1)
            ops = np.random.choice(['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'], size=7).tolist()
            ops[0] = 'input'
            ops[-1] = 'output'
            arch = {'matrix':matrix, 'ops':ops}
            if profiler.is_valid(arch):
                return Cell(matrix, ops)


    @classmethod
    def mutate(cls, model, mutate_rate, profiler):

        while True:
            new_matrix = copy.deepcopy(model.arch.matrix)
            new_ops = copy.deepcopy(model.arch.ops)

            edge_mutation_prob = mutate_rate / 7
            for src in range(0, 7 - 1):
                for dst in range(src + 1, 7):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]

            op_mutation_prob = mutate_rate / 5
            for ind in range(1, 5 + 1):
                if random.random() < op_mutation_prob:
                    available = [o for o in ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'] if o != new_ops[ind]]
                    new_ops[ind] = random.choice(available)


            arch = {'matrix': new_matrix, 'ops': new_ops}
            if profiler.is_valid(arch):
                return Cell(new_matrix, new_ops)


