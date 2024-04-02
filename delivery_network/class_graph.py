import numpy
import copy
import time
import graphviz
import random
import itertools
from Union_set import *
from Fileprio import *

class Graph:
    """
    A class representing graphs as adjacency lists and implementing various algorithms on the graphs. Graphs in the class are not oriented.
    Attributes:
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionary that contains the adjacency list of each node in the form
        graph[node] = [(neighbor1, p1, d1), (neighbor1, p1, d1), ...]
        where p1 is the minimal power on the edge (node, neighbor1) and d1 is the distance on the edge
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges.
    """

    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output

    def add_edge(self, node1, node2, power_min, dist=1):
        self.graph[node1] = self.graph[node1] + [(node2, power_min, dist)]
        self.graph[node2] = self.graph[node2] + [(node1, power_min, dist)]
        self.nb_edges += 1

    """
    Connected Components of a Graph
    Depth-First Search
    """

    def connected_components_set(self):
        n = self.nb_nodes
        """We assume that the node 0 never exists"""
        to_visit = [False] * (n + 1)
        comp = []

        def dfs(u):
            stack = [u]
            component = [u]
            to_visit[u] = True
            while len(stack) > 0:
                neighbors = self.graph[stack[-1]]
                stack.pop()
                for neighbor in neighbors:
                    if not to_visit[neighbor[0]]:
                        stack.append(neighbor[0])
                        to_visit[neighbor[0]] = True
                        component.append(neighbor[0])
            return component
        
        for i in range(1, n + 1):
            if not to_visit[i]:
                comp.append(frozenset(dfs(i)))
        return set(comp)

    """
    Before creating the get_path_with_power function, we create sub-functions
    that will allow us to simplify the code. We will then code the functions
    adj and power related to the Graph class and a function include independent of the
    Graph class.
        The first adj function takes as argument the number of a node
    and returns the list of numbers of all adjacent nodes.
        The power function takes as argument two supposed adjacent nodes, and returns
    the minimal power needed to go from one node to the other.
        The include function (at the bottom of the code) takes as argument two lists and returns
    True if the elements of the first list are included in the second list, and otherwise it
    returns an element belonging to the first list but not to the second.
    """

    def adj(self, i):
        graph = copy.deepcopy(self.graph)
        adj = []
        for node in graph[i]:
            adj.append(node[0])
        return adj

    def power(self, i, j):
        graph = self.graph
        for node in graph[i]:
            if node[0] == j:
                return node[1]

    def get_path_with_power(self, src, dest, p, visited=None, real_visited=None, total_power=0):
        if visited is None:
            """The visited list will contain all nodes traversed during recursion"""
            visited = [src]

        if real_visited is None:
            """The real_visited list will contain only the nodes of the path being explored. 
            The list is therefore included in visited"""
            real_visited = [src]

        adj_set = self.connected_components_set()
        """First, check if a path exists between source and destination"""
        for s in adj_set:
            if src in s and dest not in s:
                return None

        if src == dest:
            """When a possible path is found, distinguish between two cases: when the truck's power is sufficient and when it is not."""
            if p >= total_power:
                return real_visited, total_power
            else:
                """If the power of the current path is higher than the maximum power of the truck, then return two nodes back
                to avoid looping back on the same path. We also assume that the minimal power between two adjacent nodes
                is the power linking these two nodes."""
                if len(real_visited) > 2:
                    new_total_power = total_power - \
                        self.power(real_visited[-1], real_visited[-2]) - \
                        self.power(real_visited[-2], real_visited[-3])
                    real_visited.pop()
                    real_visited.pop()
                    return self.get_path_with_power(real_visited[-1], dest, p, visited, real_visited, new_total_power)
                else:
                    return "Not enough power"

        """If the new node is not the destination to reach, then look at which nodes have not been traversed
        in order to continue recursion."""
        adj = self.adj(src)
        if Graph.include(adj, visited) == True:
            if len(real_visited) == 1:
                return None
            else:
                real_visited.remove(src)
                new_total_power = total_power - self.power(src, real_visited[-1])
                return self.get_path_with_power(real_visited[-1], dest, p, visited, real_visited, new_total_power)
            
            """If all nodes adjacent to the source have been traversed, then backtrack on the path in order to
            find another branch leading to the destination.
            """
        else: 
            new_src = Graph.include(adj, visited)
            new_visited = visited + [new_src]
            new_real_visited = real_visited + [new_src]
            new_total_power = total_power + self.power(src, new_src)
            return self.get_path_with_power(new_src, dest, p, new_visited, new_real_visited, new_total_power)
    



    """
    Min_power 1
    """

    """
    We can now write the min_power function which will optimize the route from src to dest
    by minimizing the power required for the journey. This function takes the source and destination as arguments
    and returns the route and the minimal power.
    """

    """ 
    To optimize the network, we need to create a function that returns the route between two nodes minimizing
    the power required. For this, we use the same code as before but by modifying a few lines,
    in particular the fact that the function will stop when it has traversed the entire graph.
    """

    def min_power(self, src, dest, p_min=None, visited=None, real_visited=None, p=None, t_min=[]):
        # Initialize default values if not provided
        if p_min == None:
            p_min = numpy.inf

        if p == None:
            p = [0]

        if visited == None:
            """The visited list will contain all nodes traversed during recursion"""
            visited = [src]

        if real_visited == None:
            """The real_visited list will contain only the nodes of the path being explored. 
            The list is therefore included in visited"""
            real_visited = [src]

        # Get the connected components of the graph
        adj_set = self.connected_components_set()
        """First, check if a path exists between source and destination"""
        for s in adj_set:
            if src in s and dest not in s:
                return None

        # Base case: If source is equal to destination
        if src == dest:
            """When a possible path is found, distinguish between two cases: when the truck's power is sufficient and when it is not."""
            if p_min > p[-1]:
                p_min = p[-1]
                t_min = real_visited

            else:
                """If the power of the current path is higher than the maximum power of the truck, then return two nodes back
                to avoid looping back on the same path. We also assume that the minimal power between two adjacent nodes
                is the power linking these two nodes."""
                if len(real_visited) > 2:
                    if self.power(real_visited[-1], real_visited[-2]) == p[-1]:
                        p.pop()
                    if self.power(real_visited[-2], real_visited[-3]):
                        p.pop()
                    real_visited.pop()
                    real_visited.pop()
                    print(real_visited, p)
                    return(self.min_power(real_visited[-1], dest, p_min, visited, real_visited, p, t_min))

                else:
                    return ("Not enough power")

        """If the new node is not the destination to reach, then look at which nodes have not been traversed
        in order to continue recursion."""
        adj = self.adj(src)
        if Graph.include(adj, visited) == True:
            if len(real_visited) == 1:
                return (t_min, p_min)
            else:
                if p[-1] == self.power(real_visited[-1], real_visited[-2]):
                    p.pop()
                real_visited.remove(src)
                print('Retour', visited, real_visited, p)
                return(self.min_power(real_visited[-1], dest, p_min, visited, real_visited, p, t_min))

            """If all nodes adjacent to the source have been traversed, then backtrack on the path in order to
            find another branch leading to the destination."""
        else:
            new_src = Graph.include(adj, visited)
            new_visited = visited + [new_src]
            new_real_visited = real_visited + [new_src]
            if self.power(src, new_src) >= p[-1]:
                p.append(self.power(src, new_src))
            print(new_visited, new_real_visited, p)
            return(self.min_power(new_src, dest, p_min, new_visited, new_real_visited, p, t_min))

    def min_power1(self, i, j, t=[], p=None):
        # Set default value for power if not provided
        if p == None:
            p = numpy.inf
        
        # Get the path with power between nodes i and j
        path = self.get_path_with_power(i, j, p)
        # If no path is found, return the current path and power
        if path == None:
            return t, p
        else:
            # Find the minimum power path between nodes i and j
            return self.min_power(i, j, path[0], path[1]-1)


    """
    Min_power 2
    """

    def dijikstra(self, src):
        # Initialize variables
        n = self.nb_nodes
        d = [numpy.inf]*n  # Distance list
        h = [False]*n  # Visited list
        F = Fileprio(0, n)  # Priority queue
        f = [False]*n  # Visited flag list
        prec = [i for i in range(n)]  # Predecessor list
        Fileprio.enfiler(F, src-1, 0)  # Enqueue the source node with priority 0
        f[src-1] = True
        d[src-1] = 0

        # Dijkstra's algorithm
        while F.n > 0:
            u = Fileprio.supprimer_min_fp(F)  # Remove node with minimum priority from priority queue
            l = self.graph[u+1]  # Get neighbors of node u
            for v in l:
                if (not f[v[0]-1]) and (not h[v[0]-1]):
                    Fileprio.enfiler(F, v[0]-1, d[v[0]-1])  # Enqueue node v with priority d[v]
                    f[v[0]-1] = True  # Mark node v as visited
                if d[v[0]-1] > max(d[u], v[1]):
                    # Update distance and priority if a shorter path is found
                    d[v[0]-1] = max(d[u], v[1])
                    F.poids[v[0]-1] = max(d[u], v[1])
                    prec[v[0]-1] = u  # Update predecessor

            h[u] = True  # Mark node u as visited

        return prec, d  # Return predecessors and distances


    def min_power2(self, src, dest):
        # Find shortest path using Dijkstra's algorithm
        prec, d = Graph.dijikstra(self, src)
        l = [dest]  # Initialize list with destination node
        u = dest-1  # Start with destination node
        # Traverse predecessors to find the shortest path
        while u != (src-1):
            u = prec[u]
            l.append(u+1)
        l.reverse()  # Reverse the path to get it from source to destination
        return l, d[dest-1]  # Return the shortest path and its power


    """
    Min_power 3
    """

    """
    For get_path_with_power, we will implement an algorithm to determine the minimum power for every possible path using the Floyd-Warshall algorithm.
    """

    def edge_i_j(self, i, j):
        """
        Returns the power (weight) of the edge between nodes i and j.
        """
        l = self.graph[i]
        for k in range(len(l)):
            if l[k][0] == j:
                return(l[k][1])
        return(numpy.inf)

    def matrice_adj(self):
        """
        Constructs the adjacency matrix of the graph.
        """
        n = self.nb_nodes
        m = numpy.array([[numpy.inf]*n]*n)
        for i in range(n):
            for j in range(n):
                m[i][j] = Graph.edge_i_j(self, i+1, j+1)
        return(m)

    def matrice_pi(self):
        """
        Constructs the predecessor matrix of the graph.
        """
        m = Graph.matrice_adj(self)
        n = len(m)
        pi = numpy.array([[numpy.inf]*n]*n)
        for i in range(n):
            for j in range(n):
                if m[i][j] < numpy.inf:
                    pi[i][j] = j
        return(pi)

    def floydwarshall(self):
        """
        Applies the Floyd-Warshall algorithm to find the shortest paths between all pairs of nodes.
        """
        n = self.nb_nodes
        mat = Graph.matrice_adj(self)
        m = mat.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    m[i][j] = min(max(m[i][k], m[k][j]), m[i][j])
        return(m)

    def plus_court_chemin(self):
        """
        Computes the shortest path matrix and predecessor matrix.
        """
        n = self.nb_nodes
        mat = Graph.matrice_adj(self)
        pi = Graph.matrice_pi(self)
        m = mat.copy()
        c = pi.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if max(m[i][k], m[k][j]) < m[i][j]:
                        c[i][j] = c[i][k]
                    m[i][j] = min(max(m[i][k], m[k][j]), m[i][j])
        return(m, c)

    def chemin_pi(self, i, j):
        """
        Computes the shortest path between nodes i and j using the predecessor matrix.
        """
        m, pi = Graph.plus_court_chemin(self)
        power = m[i-1][j-1]
        l = [i]
        j0 = j-1
        i0 = i-1
        while i0 != j0:
            i0 = int(pi[i0][j0])
            l.append(i0+1)
        return(l, int(power))

    def get_path_with_power2(self, i, j, p):
        """
        Finds a path from node i to node j with power less than or equal to p.
        """
        l, power = Graph.chemin_pi(self, i, j)
        if p < power:
            return(None)
        else:
            return(l)

    def min_power3(self, i, j):
        """
        Computes the minimum power path between nodes i and j.
        """
        return(Graph.chemin_pi(self, i, j))


    def edge_i_j_dist(self, i, j):
        """
        Returns the distance (weight) of the edge between nodes i and j.
        """
        l = self.graph[i]
        for k in range(len(l)):
            if l[k][0] == j:
                return(l[k][2])
        return(numpy.inf)


    def visualisation_graphe(self):
        """
        Draws the graph with the weight and the distance of each edges
        Parameters: 
        -----------
        t: list
        represents the traject to draw
        """
        dot = graphviz.Graph('graphe', comment='Le Graphe')
        g = self.graph
        ar, = []
        for i in self.nodes:
            dot.node(str(i), str(i))
            l = g[i]
            for j in l:
                if ({i, j[0]} not in ar):
                    dot.edge(str(i), str(
                        j[0]), "weight = {} \n distance = {} ".format(j[1], j[2]))
                    ar.append({i, j[0]})
        dot.view()


    def visualisation_graphe_chemin(self, t):
        """
        Draws the graph and highlights the route t with the weight and the distance of each edges.

        Parameters:
        -----------
        t : list
            The list containing the nodes representing the path.

        Returns:
        --------
        None
        """
        dot = graphviz.Graph('graphe', comment='Le Graphe')
        g = self.graph
        n = len(t)
        ar, no, c = [], [t[-1]], []
        dot.node(str(t[0]), str(t[0]), color="red", fontcolor="red")
        for i in range(n-1):
            c.append({t[i], t[i+1]})
        for i in self.nodes:
            if (not i in no) and (i in t):
                dot.node(str(t[i]), str(t[i]), color="red", fontcolor="red")
            if not i in no:
                dot.node(str(i), str(i))
            l = g[i]
            for j in l:
                if ({i, j[0]} not in ar) and ({i, j[0]} in c):
                    dot.edge(str(i), str(j[0]), "weight = {} \n distance = {} ".format(
                        j[1], j[2]), color="red", fontcolor="red")
                    ar.append({i, j[0]})
                if ({i, j[0]} not in ar):
                    dot.edge(str(i), str(
                        j[0]), "weight = {} \n distance = {} ".format(j[1], j[2]))
                    ar.append({i, j[0]})
        dot.view()


    def tree(self, root=None):
        """
        Constructs the tree starting from the root node.

        Parameters:
        -----------
        root : node, optional
            The root node of the tree. If not provided, the first node in the graph is considered as the root.

        Returns:
        --------
        parent : dict
            A dictionary containing the parent of each node in the tree.
        depths : dict
            A dictionary containing the depth of each node in the tree.
        """
        if root == None:
            root = self.nodes[0]
        graph = self.graph
        visited = set()
        parent = {root: None}
        depths = {}
        stack = [root]
        depths[root] = 0
        while stack:
            node = stack.pop()
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor[0] not in visited:
                    parent[neighbor[0]] = (node, self.power(node, neighbor[0]))
                    depths[neighbor[0]] = depths[node] + 1
                    stack.append(neighbor[0])
        return parent, depths


    def commun_ancestor(self, node1, node2, parents=None, depths=None):
        """
        Returns the minimal power of the truck to have in order to go from node1 to node2.

        Parameters:
        -----------
        node1 : node
            The starting node.
        node2 : node
            The destination node.
        parents : dict, optional
            A dictionary with the parent of each key in a tree predefined.
        depths : dict, optional
            A dictionary with the depth of each node in a tree predefined.

        Returns:
        --------
        node : node
            The common ancestor node.
        min_power : int
            The minimal power required to travel from node1 to node2.
        """
        min_power = 0
        if parents == None:
            parents, depths = self.tree(node1)
        while node1 != node2:
            if depths[node1] >= depths[node2]:
                power = self.power(node1, parents[node1][0])
                if power > min_power:
                    min_power = power
                node1 = parents[node1][0]
            elif depths[node2] > depths[node1]:
                power = self.power(node2, parents[node2][0])
                if power > min_power:
                    min_power = power
                node2 = parents[node2][0]
        return node2, min_power

          