#!/usr/bin/python3 

import cProfile as profile
import math
import numpy as np
import random
import soundfile as sf
import sys

from tqdm import tqdm, trange

def aggregate(lst):
    return sum(lst)

def activate(value):
#    return 1. / (1. + np.exp(-value)) # SIGMOID
    return np.tanh(value)

def distance(g1, g2):
    return abs(g1.value() - g2.value())

class Gene(object):
    last_id = -1

    @classmethod
    def next_id(cls):
        cls.last_id += 1
        return cls.last_id

    def __init__(self, generation, key=None):
        if key is None:
            key = self.next_id()
        self.key = key
        self.generation = generation

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def mutate(other):
        pass

    def crossover(self, other):
        return self.__class__(self.key)


class NodeGene(Gene):
    def __init__(self, generation, aggregation_fn, activation_fn, bias=0, response=1, key=None):
        super(NodeGene, self).__init__(generation, key=key)
        self.agg = aggregation_fn
        self.act = activation_fn
        self.bias     = bias
        self.response = response
        self.generation = generation

    def aggregate(self, values):
        return self.agg(values)

    def activate(self, value):
        return self.act(value)

    def __repr__(self):
        return '[Node #{}]'.format(self.key)

    def value(self):
        return self.key + self.bias + self.response

    def copy(self):
        return NodeGene(
            self.generation,
            self.agg,
            self.act,
            self.bias,
            self.response,
            self.key
        )

class ConnGene(Gene):
    def __init__(self, generation, src, dst, weight=1, key=None, enabled=True):
        super(ConnGene, self).__init__(generation, key=key)
        self.generation = generation
        self.src     = src
        self.dst     = dst
        self.weight  = weight
        self.enabled = enabled

    def __repr__(self):
        return '[Conn #{}]'.format(self.key)

    def value(self):
        return (self.dst.key - self.src.key) + self.weight + int(self.enabled)

    def copy(self):
        return ConnGene(
            self.generation,
            self.src,
            self.dst,
            self.weight,
            self.key,
            self.enabled
        )

class Genome(object):
    last_id = -1

    @classmethod
    def next_id(cls):
        cls.last_id += 1
        return cls.last_id

    def __init__(self, generation, inputs, outputs, key=None, uninitialized=False, fully_connected_start=True):
        if key is None:
            key = self.next_id()
        self.key     = key
        self.inputs  = inputs
        self.outputs = outputs
        self.generation = generation
        self.invalid    = True

        self.nodes  = {}
        if not uninitialized:
            for i in range(inputs + outputs):
                self.add_node(generation, -i-1)

        self.conns   = {}
        if not uninitialized and fully_connected_start:
            for i in range(inputs):
                for o in range(outputs):
                    self.add_conn(generation, self.nodes[-i-1], self.nodes[-inputs-o-1])
        self.fitness = -float('inf')
                 
    def value(self):
        value = 0
        for gene in self.conns.values():
            value += gene.value()
        for gene in self.nodes.values():
            value += gene.value()
        return value

    def add_node(self, generation, key=None):
        new_node = NodeGene(generation, aggregate, activate, key=key)
        self.nodes[new_node.key] = new_node
        return new_node

    def add_conn(self, generation, n1, n2, weight=1.0, key=None):
        new_conn = ConnGene(generation, n1, n2, weight=weight, key=key)
        self.conns[new_conn.key] = new_conn

    def __repr__(self):
        return '[Genome #{}: {} nodes, {} connections. Fitness: {}]'.format(
            self.key, len(self.nodes), len(self.conns), self.fitness)

    def mutate(self, generation, node_add_chance=0.05, node_remove_chance=0.01,
        conn_add_chance=0.1, conn_remove_chance=0.05
    ):
        if np.random.random() < node_add_chance:
            self.mutate_add_node(generation)
        if np.random.random() < node_remove_chance:
            self.mutate_remove_node(generation)
        if np.random.random() < conn_add_chance:
            self.mutate_add_conn(generation)
        if np.random.random() < conn_remove_chance:
            self.mutate_remove_conn(generation)

        for node in self.nodes.values():
            node.mutate()

        for conn in self.conns.values():
            conn.mutate()

    def mutate_add_node(self, generation):
        if len(self.conns) == 0:
            return

        conn = random.choice(list(self.conns.values()))
        conn.enabled = False # To be collected eventually

        new_node = self.add_node(generation)
        self.add_conn(generation, conn.src, new_node, 1.0)
        self.add_conn(generation, new_node, conn.dst, conn.weight)

    def mutate_remove_node(self, generation):
        pass

    def mutate_add_conn(self, generation):
        possible_inputs  = []
        possible_outputs = []
        for key in self.nodes:
            if key >= -self.inputs: # Removes Output nodes
                possible_inputs.append(key)
            if key >= 0 or key <= -self.inputs-1: # Removes Input nodes
                possible_outputs.append(key)

        ikey = random.choice(possible_inputs)
        okey = random.choice(possible_outputs)

        if ikey < -self.inputs-1: # Outputs can't be inputs.
            return

        if self.creates_cycles(ikey, okey): # We're feeding forward for now.
            return

        inode = self.nodes[ikey]
        onode = self.nodes[okey]

        for conn in self.conns.values():
            if conn.src == inode and conn.dst == onode: # Avoid duplication
                conn.enabled = True
                return

        self.add_conn(generation, inode, onode)

    def mutate_remove_conn(self, generation):
        pass

    def creates_cycles(self, ikey, okey):
        return False

    def crossover(self, other, generation):
        assert self.inputs  == other.inputs
        assert self.outputs == other.outputs

        assert self.fitness  is not None
        assert other.fitness is not None

        if self.fitness > other.fitness:
            parent1 = self
            parent2 = other
        else:
            parent1 = other
            parent2 = self

        child = Genome(generation, self.inputs, self.outputs, uninitialized=True)

        for key, conn1 in parent1.conns.items():
            try:
                conn2 = parent2.conns[key]
                child.conns[key] = conn1.crossover(cross2)
            except:
                child.conns[key] = conn1.copy()

        for key, node1 in parent1.nodes.items():
            try:
                node2 = parent2.nodes[key]
                child.nodes[key] = node1.crossover(node2)
            except:
                child.nodes[key] = node1.copy()

        return child


    def find_required_nodes(self, inputs, outputs):
        required = set(outputs)
        current  = set(outputs)

        while True:
            nodes = set()
            for conn in self.conns.values():
                if conn.dst in current and conn.src not in current:
                    nodes.add(conn.src)
            
            if len(nodes) == 0:
                break

            layer_nodes = set()
            for node in nodes:
                if node not in inputs:
                    layer_nodes.add(node)

            if len(layer_nodes) == 0:
                break

            required = required.union(layer_nodes)
            current  = current.union(nodes)

        return required


    def feed_forward(self):
        if not self.invalid:
            return self.network

        enabled = [ conn for conn in self.conns.values() if conn.enabled ]

        input_cache  = {}
        output_cache = {}
        for conn in enabled:
            try:
                input_cache[conn.dst.key].append(conn)
            except:
                input_cache[conn.dst.key] = [conn]
            try:
                output_cache[conn.src.key].append(conn)
            except:
                output_cache[conn.src.key] = [conn]

        network = Network(self, self.inputs, self.outputs)
        current = set()
        for i in range(self.inputs):
            for conn in output_cache[-i-1]:
                current.add(conn.dst)

        while len(current) > 0:
            next_set = set()
            for node in current:
                network.add(node, input_cache[node.key])
                if node.key < -self.inputs:
                    continue
                for conn in output_cache[node.key]:
                    next_set.append(conn.dst)

            current = next_set

        self.invalid = False
        self.network = network
        return network

class Network(object):
    def __init__(self, genome, inputs, outputs):
        self.genome  = genome
        self.inputs  = inputs
        self.outputs = outputs
        self.values = { -i-1: 0.0 for i in range(inputs + outputs) }
        self.steps = []

    def add(self, node, inputs):
        self.steps.append((node, inputs))

    def compute(self, inputs):
        if self.inputs != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.inputs), len(inputs)))

        for i, value in zip(range(self.inputs), inputs):
            self.values[-i-1] = value

        for node, incoming in self.steps:
            nodes = []
            for conn in incoming:
                value  = self.values[conn.src.key]
                weight = conn.weight
                nodes.append(value * weight)
            value = node.bias + node.response * node.aggregate(nodes)
            self.values[node.key] = node.activate(value)
        return [ self.values[-(self.inputs + i)-1] for i in range(self.outputs) ]

class Species:
    last_id = -1

    @classmethod
    def next_id(cls):
        cls.last_id += 1
        return cls.last_id

    def __init__(self, generation, key=None):
        if key is None:
            key = self.next_id()
        self.key     = key
        self.genomes = []

        self.generation    = generation
        self.last_improved = generation
        self.best_fitness  = None
        self.representative = None

        self.sum_fitness   = 0
        self.mean_fitness  = 0
        self.min_fitness   = 0
        self.max_fitness   = 0

    def __len__(self):
        return len(self.genomes)

    def add(self, genome):
        self.genomes.append(genome)

    def update(self, generation):
        self.representative = self.genomes[0]

        self.fitness = 0
        self.min_fitness = self.genomes[0].fitness
        self.max_fitness = self.min_fitness
        for genome in self.genomes:
            self.sum_fitness += genome.fitness
            if genome.fitness < self.min_fitness:
                self.min_fitness = genome.fitness
            elif genome.fitness > self.max_fitness:
                self.max_fitness = genome.fitness
                self.representative = genome
        self.mean_fitness = self.sum_fitness / len(self.genomes)

        self.genomes.sort(key=lambda g: g.fitness)

        if self.best_fitness is None or self.mean_fitness > self.best_fitness:
            self.best_fitness  = self.mean_fitness
            self.last_improved = generation

class Population:
    def __init__(self, size):
        # TODO: Symbiosis
        self.enc_genus  = [Species(0)]
        self.dec_genus  = [Species(0)]
        self.crit_genus = [Species(0)]
                          
        for _ in range(size):
            self.enc_genus[0].add(Genome(0, 512, 51))
            self.dec_genus[0].add(Genome(0, 51, 512))
            self.crit_genus[0].add(Genome(0, 512, 1))

        self.size = size

        self.enc_best_fit  = None
        self.enc_best_gen  = None
        self.dec_best_fit  = None
        self.dec_best_gen  = None
        self.crit_best_fit = None
        self.crit_best_gen = None

    def __repr__(self):
        return '[Population: {}, {}, {}]'.format(len(self.enc_genus), len(self.dec_genus), len(self.crit_genus))

    def _all_genomes(self, genus, reset_fit=False):
        for species in genus:
            for genome in species.genomes:
                if reset_fit:
                    genome.fitness = 0
                yield genome

    def evaluate(self, inputs):
        enc_nets  = [ genome.feed_forward() for genome in self._all_genomes(self.enc_genus,  reset_fit=True) ]
        dec_nets  = [ genome.feed_forward() for genome in self._all_genomes(self.dec_genus,  reset_fit=True) ]
        crit_nets = [ genome.feed_forward() for genome in self._all_genomes(self.crit_genus, reset_fit=True) ]
        
        assert len(enc_nets) > 0
        assert len(dec_nets) > 0
        assert len(crit_nets) > 0

        sum_gen_fit = 0
        sum_dis_fit = 0

        for enet, dnet, cnet in zip(enc_nets, dec_nets, crit_nets):
            values = enet.compute(inputs)
            values = dnet.compute(values)

            real_dist = np.linalg.norm(inputs - values)

            real_loss = cnet.compute(inputs)[0]
            fake_loss = cnet.compute(values)[0]

            gen_fitness = -abs(real_dist - fake_loss)
            dis_fitness = -abs(real_loss) + gen_fitness
            
            cnet.genome.fitness = dis_fitness
            enet.genome.fitness = gen_fitness
            dnet.genome.fitness = gen_fitness

            sum_dis_fit += dis_fitness
            sum_gen_fit += gen_fitness

            if self.enc_best_fit is None or gen_fitness > self.enc_best_fit:
                self.enc_best_fit = gen_fitness
                self.enc_best_gen = enet.genome

            if self.dec_best_fit is None or gen_fitness > self.dec_best_fit:
                self.dec_best_fit = gen_fitness
                self.dec_best_gen = dnet.genome

            if self.crit_best_fit is None or dis_fitness > self.crit_best_fit:
                self.crit_best_fit = dis_fitness
                self.crit_best_gen = cnet.genome


        return sum_gen_fit / len(enc_nets), sum_dis_fit / len(crit_nets)
        
    def _reproduce(self, generation, genus, max_stagnation=200, min_species_size=10,
        elitism=2, survival_threshold=0.5
    ):
        min_fit = None
        max_fit = None
        sum_fit = 0
        cnt_fit = 0

        remaining  = []
        prev_sizes = []
        for species in genus:
            species.update(generation)
            if generation - species.last_improved < max_stagnation:
                remaining.append(species)
                prev_sizes.append(len(species))

            sum_fit += species.sum_fitness
            cnt_fit += len(species)

            if min_fit is None or species.min_fitness < min_fit:
                min_fit = species.min_fitness
            if max_fit is None or species.max_fitness > max_fit:
                max_fit = species.max_fitness

        if len(remaining) == 0:
            raise Exception('Extinction')

        range_fit    = max(0.0001, max_fit - min_fit)
        adjusted_sum = 0
        for species in remaining:
            adjusted = (species.mean_fitness - min_fit) / range_fit
            species.adjusted_fitness = adjusted
            adjusted_sum += adjusted
    
        min_species_size = max(min_species_size, elitism)
        
        spawn_sizes = []
        spawn_sum   = 0
        for i, size in enumerate(prev_sizes):
            if adjusted_sum > 0:
                s = max(min_species_size, remaining[i].adjusted_fitness / adjusted_sum * self.size)
            else:
                s = min_species_size

            d = (s - size) * 0.5
            c = int(round(d))
            spawn = size
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1
            spawn_sum += spawn
            spawn_sizes.append(spawn)
        
        norm = self.size / spawn_sum
        spawn_sizes = [ max(min_species_size, int(round(n * norm))) for n in spawn_sizes ]

        new_pop = []
        for size, species in zip(spawn_sizes, remaining):
            spawn = max(spawn, elitism)
            assert spawn > 0

            old_genomes = species.genomes
            species.genomes = []
            
            for i in range(elitism):
                species.genomes.append(old_genomes[-i-1])
                spawn -= 1

            if spawn <= 0:
                continue

            cutoff = int(math.ceil(survival_threshold * len(old_genomes)))
            cutoff = max(cutoff, 2)

            old_genomes = old_genomes[:cutoff]

            while spawn > 0:
                spawn -= 1

                parent1, parent2 = np.random.choice(old_genomes, 2)
                child = parent1.crossover(parent2, generation)
                child.mutate(generation)
                species.genomes.append(child)

            new_pop.append(species)

        return new_pop

    def reproduce(self, generation):
        self.enc_genus  = self._reproduce(generation, self.enc_genus)
        self.dec_genus  = self._reproduce(generation, self.dec_genus)
        self.crit_genus = self._reproduce(generation, self.crit_genus)

    def _speciate(self, generation, genus, compatibility_threshold=0.5):
        assert len(genus) > 0

        unspeciated = list(self._all_genomes(genus))

        min_value =  float('inf')
        max_value = -float('inf')

        for species in genus:
            if species.representative is None:
                species.representative = species.genome[0]

            candidate = (float('inf'), None)
            for genome in unspeciated:
                gen_dist  = distance(species.representative, genome)
                gen_value = genome.value()
                if gen_dist < candidate[0]:
                    candidate = (gen_dist, genome)
                if gen_value < min_value:
                    min_value = gen_value
                if gen_value > max_value:
                    max_value = gen_value
            species.representative = candidate[1]
            species.genomes        = [candidate[1]]
            unspeciated.remove(candidate[1])

        range_value = max_value - min_value
        if range_value == 0:
            range_value = 1

        while len(unspeciated) > 0:
            genome    = unspeciated.pop()
            candidate = (float('inf'), None)

            for species in genus:
                gen_dist   = distance(species.representative, genome)
                similarity = gen_dist / range_value

                if similarity < compatibility_threshold and gen_dist < candidate[0]:
                    candidate = (gen_dist, species)

            if candidate[1] is not None:
                species.genomes.append(genome)
            else:
                new_species = Species(generation)
                new_species.representative = genome
                new_species.genomes        = [genome]
                genus.append(new_species)

        for species in genus:
            if len(species.genomes) < 2:
                genus.remove(species)

        return genus

    def speciate(self, generation):
        self.enc_genus  = self._speciate(generation, self.enc_genus)
        self.dec_genus  = self._speciate(generation, self.dec_genus)
        self.crit_genus = self._speciate(generation, self.crit_genus)

def main():
    population = Population(150)

    try:
        blocks      = sf.blocks('inputs/hello_world.wav', blocksize=512, overlap=32)
        generations = tqdm(range(1))
        for generation in generations:
            block = next(blocks)
            block = np.pad(block, (0, 512 - len(block)), 'constant')

            fitness = population.evaluate(block)
            population.reproduce(generation)
            population.speciate(generation)

            generations.set_postfix({ 'G': '{:.4e}'.format(fitness[0]), 'D': '{:.4e}'.format(fitness[1]) })
    except AssertionError:
        pass
    print(population)
    print(population.enc_best_gen, population.enc_best_fit) 
    print(population.dec_best_gen, population.dec_best_fit) 
    print(population.crit_best_gen, population.crit_best_fit) 
    print()

import flamegraph
flamegraph.start_profile_thread(fd=open("./perf.log", "w"))
main()
