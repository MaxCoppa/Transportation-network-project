# Transportation network project

Through this programming project, we will focus on optimizing a transportation network. The problem statement is as follows: Consider a road network consisting of cities and routes between the cities. The objective of the project will be to construct a delivery network capable of covering a set of routes between two cities with trucks. The difficulty lies in the fact that each route has a minimum power requirement, meaning a truck can only use this route if its power is greater than or equal to the route's minimum power requirement. Therefore, we need to determine for each pair of cities if a truck with a given power can find a possible path between these two cities; then, optimize the fleet of trucks to be purchased based on the routes to be covered.

The road network is represented by an undirected graph G = (V, E). The set of vertices V corresponds to the set of cities. The set of edges E corresponds to the set of existing routes between two cities. Each edge e in E is associated with a value (or weight) p ≥ 0, which is called the minimum power of the edge e, representing the minimum power required for a truck to pass through route e. Each edge e in E is also associated with a second value > 0 representing the distance between the two vertex cities on edge e.

We consider a set T of routes, where each route t is a pair of two distinct cities, i.e., t = (v, v0) where v ∈ V, v0 ∈ V. The set T represents the set of city pairs (v, v0) for which we want to have a truck that can transport between v and v0. Note that the graph is undirected and we do not distinguish the direction of the route. Each route t is also associated with a profit (or utility) ut ≥ 0, which will be earned by the delivery company if route t is covered.

Finally, the transportation is done by trucks. Each truck (denoted as K) has a power p and costs a price c. Transport on a route t = (v, v0) in T will be possible if and only if we can find in the graph G a path from v to v0 where the minimum power of each edge is less than or equal to p (i.e., the truck has sufficient power to pass everywhere on the path). It is then said that route t can be covered by the truck K considered.

To best address the problem, which is to maximize a company's profit based on the profit brought by each route and the cost of each truck depending on its power, we proceeded in two main parts: the implementation of algorithms to find the minimum power to travel a route; and the optimization of truck acquisition based on a list of routes associated with a profit.




# Delivery_network

 In this folder, there are several Python files that contain implementations of various algorithms and classes related to graph theory and optimization problems. Below is a description of each file:

1. class_graph.py:
   - This file contains the class Graph
   - This class provides functionalities to represent and manipulate graphs, edges, and nodes.

2. Fileprio.py:
   - This file includes a class named FilePriority.
   - The FilePriority class implements priority queue operations used in certain graph algorithms.

3. Union_set.py:
   - This file contains the Union_set class.
   - The Union_set class represents disjoint sets and implements various algorithms related to union-find operations.

4. graph.py:
   - This file consists of implementations of the Floyd-Warshall and Kruskal algorithms for calculating distances and shortest path in graphs.
   - The Floyd-Warshall algorithm finds the shortest paths between all pairs of vertices in a weighted graph.
   - The Kruskal algorithm finds the minimum spanning tree of a connected, undirected graph.

5. knapsack.py:
   - This file provides solutions to the knapsack problem using naive and greedy algorithms.
   - The solutions aim to maximize the profit of selecting items with given weights and values, subject to a weight constraint.

6. knapsack_genetic.py:
   - This file offers a solution to the knapsack problem using a genetic algorithm.
   - Genetic algorithms are evolutionary algorithms inspired by natural selection processes that aim to find optimal solutions to optimization problems.

7. main.py:
   - The main file serves as an entry point for running and testing the implemented algorithms and classes.
   - It may include sample usage of the provided functionalities and algorithms

# Input Files Format

The input folder contains 2 types of files: the network.x.in files (x ranging from 00 to 10) containing the graphs and the routes.x.in files (x ranging from 1 to 10) containing sets of routes for the corresponding graph x.

The structure of the network.x.in files is as follows:

- The first line consists of two integers separated by a space: the number of vertices (n) and the number of edges (m).
- The following m lines each represent an edge and are composed of 3 or 4 numbers separated by spaces: `ville1 ville2 puissance [distance]`, where `ville1` et `ville2` are the vertices of the edge, 'power' is the minimum power required to pass through the edge, and 'distance' (optional) is the distance between `ville1` et `ville2` on the edge.
The structure of the routes.x.in files is as follows:

The first line contains an integer corresponding to the number of routes in the set (T).
The following T lines each contain a route in the form `ville1 ville2 utilité`, where '`utilité`' is the profit earned if the corresponding route is covered.
