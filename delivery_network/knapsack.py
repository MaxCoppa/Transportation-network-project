
from graph import *
from class_graph import *

def combinations(E, p):
    """Returns a list of p-combinations of elements from E."""
    n = len(E)
    result = []  # List to store the combinations
    pos = list(range(p))  # Initially, take all the leftmost positions
    
    while pos[0] <= n - p:  # Loop until the leftmost position reaches the last valid combination
        result.append(tuple(E[a] for a in pos))  # Add the current combination to the result list
        
        j = p - 1  # Start from the rightmost position
        while pos[j] == n - p + j:  # While the j-th position is the rightmost possible
            j -= 1  # Decrease j
        pos[j] += 1  # Shift the j-th position to the right
        for k in range(j, p - 1):  # Move the other positions immediately to its left
            pos[k + 1] = pos[k] + 1
            
    return result


def knapsack_naive(g, routes, trucks, B):
    """Solves the knapsack problem using a brute-force approach."""
    g, u = kruskal(g)
    
    profit = 0  # Initialize profit
    budget = 0  # Initialize total cost
    max_profit = 0  # Initialize maximum profit found so far
    
    # Generate all possible combinations of routes
    combination_list = [combinations(([i] for i in range(len(routes))), k) for k in range(1, len(routes) + 1)]
    
    # Iterate through each combination
    for combination in combination_list:
        profit = 0  # Reset profit for each combination
        L = []  # Initialize list to store selected routes
        
        # Iterate through each route index in the combination
        for i in combination:
            start, end = routes[i][0], routes[i][1]  # Get start and end nodes of the route
            power, cost = truck_assigned(trucks, u, start, end)  # Calculate power and cost for the truck
            
            uti = routes[i][2]  # Get the utility of the route
            budget += cost  # Add cost to the total budget
            
            L.append[(power, cost), (start, end)]  # Append truck power, cost, and route to the list
            
            if budget > B:  # Check if budget exceeds the given budget
                break
            
            profit += uti  # Add route utility to profit
        
        # Update maximum profit if current combination yields higher profit
        if profit > max_profit:
            max_profit = profit
            
    return (L, max_profit)  # Return selected routes and maximum profit

        

"""
Second approach: Optimizing truck selection
"""
    
def trucks2(file_trucks):
    # Initialize an empty dictionary to store the selected trucks
    truck = {}
    
    # Iterate over each truck in the input list
    for power, cost in file_trucks:
        # Iterate over the existing trucks in reverse order to compare and update
        for power1, cost1 in list(truck.items()):
            # Check if the current truck is dominated by any existing truck
            if (power1 > power and cost > cost1) or (power > power1 and cost1 > cost):
                break  # Stop further comparison if dominated
            elif power1 > power or cost1 > cost:
                truck.pop(power1)  # Remove the dominated truck
        else:
            truck[power] = cost  # Add the current truck if not dominated
    return truck

"""
Approach 3: Optimal return of a truck for each route
"""

def truck_assigned(trucks, u, start, end):
    # Determine the minimum power required for the route
    min_power = min_power_union_set(u, start, end) 
    
    # Iterate over the available trucks
    for power, cost in trucks.items():
        # Check if the truck's power is sufficient for the route
        if power >= min_power:
            return power, cost  # Return the optimal truck
    return None  # Return None if no suitable truck is found


"""
Approach 4: Presentation and algorithms for the optimization problem: a knapsack problem
"""

def ratio(file):
    return file[1] / file[2]  # Calculate the profit-to-weight ratio

def ratio_list(u, routes, trucks):
    L = []
    for route in routes:
        start, end = route[0], route[1]
        t = truck_assigned(trucks, u, start, end)
        if t is not None:
            power, cost = t
            L.append([(start, end), route[2], cost, power])
    return L  # Return a list of route-truck combinations with their associated ratios

def greedy(g, routes, trucks, B):
    g, u = kruskal(g)  # Get the minimum spanning tree and its union set
    profit = 0
    budget = 0
    path = []
    truck = []
    ratio_sorted_list = sorted(ratio_list(u, routes, trucks), key=ratio, reverse=True)
    i = 0
    while i < len(ratio_sorted_list) and budget < B:
        pack = ratio_sorted_list[i]
        cost = pack[2]
        if cost + budget < B:
            path.append(pack[0])
            truck.append((pack[3], pack[2]))
            profit += pack[1]
            budget += cost
        i += 1
    return truck, path, profit  # Return the selected trucks, paths, and total profit

