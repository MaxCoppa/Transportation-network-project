import numpy
import copy
import time
import graphviz
import random
import itertools
from Union_set import *
from Fileprio import *
from class_graph import *


              
    
def min_power_tree(g, route):
    """
    Calculates the total time taken to find the minimal power needed for each route.

    Parameters:
    -----------
    g : Graph
        The graph object.
    route : Route
        The route object containing the routes to be evaluated.

    Returns:
    --------
    float
        The total time taken for computation.
    """
    g = kruskal(g)
    parents, depths = g.tree(g.nodes[0])
    t_tot = 0
    routes = route.graph
    visited = []
    for node in route.nodes:
        for node2 in routes[node]:
            if (node2, node) not in visited:
                visited.append((node, node2))
                t0 = time.time()
                g.commun_ancestor(routes[node][0][0], node)
                tf = time.time()
                t_tot += (tf - t0)
    return t_tot


def min_power_union_set(u, src, dest):
    """
    Finds the minimal power required to go from source to destination using Union Set.

    Parameters:
    -----------
    u : UnionSet
        The UnionSet object.
    src : int
        The source node index.
    dest : int
        The destination node index.

    Returns:
    --------
    list
        The path from source to destination.
    """
    p, _ = Union_set.path(u, src - 1, dest - 1)
    return p


def test(filename1, filename2):
    """
    Tests the functions with the provided filenames.

    Parameters:
    -----------
    filename1 : str
        The filename containing the graph data.
    filename2 : str
        The filename containing the route data.

    Returns:
    --------
    float
        The total time taken for computation.
    """
    g = graph_from_file(filename1)
    g, u = kruskal(g)
    m = route_from_file(filename2)
    n = len(m)
    t1 = time.time()
    for i in range(1, n):
        if len(m[i][0]) > 0 and len(m[i][1]) > 0:
            src, dest = int((m[i][0])), int((m[i][1]))
            src, dest = src - 1, dest - 1
            print(min_power_union_set(u, src, dest))
    t2 = time.time()
    return t2 - t1

        
def include(l1, l2):
    """ Return if the first list is included in the second one i.e all elements of the first list
    is in the second list
    Parameters :
    -----------
    l1, l2 : list
    """
    for val in l1:
        if val not in l2:
            return(val)
    return(True)


def matrice(filename):
    """
    Reads a text file and returns a matrice wich represents each "word" of the file
    Parameters: 
    -----------
    filename: str
        The name of the file
    """
    s = filename.split("\n")
    n = len(s)
    for i in range(n):
        s[i] = s[i].split(" ")
    return(s)


def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.
    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.
    Parameters: 
    -----------
    filename: str
        The name of the file
    Outputs: 
    -----------
    g: Graph
        An object of the class Graph with the graph from file_name.
    """
    s = matrice(filename)
    n, m = int(s[0][0]), int(s[0][1])
    g = Graph([i for i in range(1, n+1)])
    for j in range(1, m+1):
        node1, node2, power_min = int(s[j][0]), int(s[j][1]), int(s[j][2])
        if len(s[j]) == 4:
            dist = int(s[j][3])
        else:
            dist = 1
        Graph.add_edge(g, node1, node2, power_min, dist)
    return(g)

def route_from_file(filename):
    """
    Reads the route data from the given file and extracts the routes.

    Parameters:
    -----------
    filename : str
        The name of the file containing the route data.

    Returns:
    --------
    list
        A list of routes extracted from the file.
    """
    file = matrice(filename)
    routes = []
    n = len(file)
    for i in range(1, n):
        if len(file[i][0]) * len(file[i][0]) > 0:
            routes.append([int(file[i][0]), int(file[i][1]), int(file[i][2])])
    return routes


def truck_from_file(filename):
    """
    Reads the truck data from the given file and extracts the trucks.

    Parameters:
    -----------
    filename : str
        The name of the file containing the truck data.

    Returns:
    --------
    list
        A list of trucks extracted from the file.
    """
    file = matrice(filename)
    truck = []
    n = len(file)
    for i in range(1, n):
        if len(file[i][0]) * len(file[i][0]) > 0:
            truck.append([int(file[i][0]), int(file[i][1])])
    return truck



def graph_from_file_route(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.
    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.
    Parameters: 
    -----------
    filename: str
        The name of the file
    Outputs: 
    -----------
    g: Graph
        An object of the class Graph with the graph from file_name.
    """
    s = matrice(filename)
    n = int(s[0][0])
    g = Graph([i for i in range(1, n+1)])
    for j in range(1, n+1):
        node1, node2, power_min = int(s[j][0]), int(s[j][1]), int(s[j][2])
        if len(s[j]) == 4:
            dist = int(s[j][3])
        else:
            dist = 1
        Graph.add_edge(g, node1, node2, power_min, dist)
    return(g)

def unionset_graph(u, m):
    """
    Transform a tree implemented thanks to the Union_set Structure into a graph in the Graph class
    Parameters: 
    -----------
    u: Union_set
        The tree
    m: numpy array 
        a matrix which stock the weight of each edge in the graph

    Outputs:
    -----------
    g: Graph
        a graph associate to u
    """
    n = u.n
    l = u.p
    g = Graph([i for i in range(n)])
    for i in range(n):
        Graph.add_edge(g, i, l[i], m[i][l[i]])
    return(g)


    
def kruskal(g):
    """
    Transforms a graph into a covering tree minimum using the Union_set class.
    The complexity of this function is thank to the optimised Union_set operation in O((n+a)log(n))
    where n is the number of nodes and a the number of edges
    Parameters:
    -----------
    g: Graph
    Outputs:
    ---------
    g0: Graph
        Reresents the minimun tree covering the Graph g
    Note: We stocked the edges in a matrice to ease the operations, we could used the adjancy list with the same complexity
    but the function would be less clear. The problem of this function is a complexity in space of O(n^2) in comparaison
    ajancy list would use a space complexity in O(n)
    """
    no = g.nodes
    n = g.nb_nodes
    m0 = g.nb_edges
    l = []
    for el in no:
        adj = g.graph[el]
        for el0 in adj:
            l.append([el0[1],el-1,el0[0]-1])
    l.sort()
    u = Union_set(n)
    g = Graph([i+1 for i in range(n)])
    k,i = 0,0
    while k < n and i < 2*m0 :  
        p,x,y =int(l[i][0]), l[i][1],l[i][2]
        rx,_ = Union_set.rep(u,x)
        ry,_= Union_set.rep(u,y)
        if rx != ry:
            Union_set.fusion(u,x,y,p)
            Graph.add_edge(g,x+1, y+1, p )
            k+= 1
        i+= 1
    return(g,u)


def puissance_mini_kruskal(g, src, dest):
    """
    Returns the min_power of a path (src,dest) using the kruskal algorithm
    The complexity is in O((n+a)log(n)). Calling min_power as a complexity in O(n) because this time the graph as n edges and nodes 

    Parameters: 
    -----------
    g: Graph
    src,dest: int, int
        the path in the Graph
    """
    g,u = kruskal(g)
    p,_= Union_set.path(u, src, dest)
    return(p)
