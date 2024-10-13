# DFS Algorithm
def DFS(node, target, visited, depth):
    visited.add(node)
    print(node, end=" ")
    if node == target:
        return True
    if depth <= 0:
        return False
    for child in graph.get(node, []):
        if child not in visited:
            if DFS(child, target, visited, depth - 1):
                return True
    return False

# IDS Algorithm
def IDS(start, target, max_depth):
    for depth in range(max_depth + 1):
        visited = set()
        print(f"\nDepth level: {depth}")
        print("Visited nodes: ", end="")
        if DFS(start, target, visited, depth):
            return True
        print("\n")
    return False

graph = {
    '5' : ['3', '7'],
    '3' : ['2', '4'],
    '7' : ['8'],
    '2' : [],
    '4' : ['8'],
    '8' : []
}

start_node = '5'
target_node = '8'
max_depth = 3

goal_node = IDS(start_node, target_node, max_depth)
if goal_node:
    print(f"\nTarget node '{target_node}' found.")
else:
    print(f"\nTarget node '{target_node}' not found within depth {max_depth}.")
