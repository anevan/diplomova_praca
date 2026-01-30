import heapq

def greedy_correlation_path(matrix, start_node, end_node):
    """
       Greedy pathfinding over a correlation matrix.
       At each step, it selects the unvisited attribute most strongly correlated (in absolute value)
       with the current node, and stops when the end node is reached.

       Note: Greedy is shortsighted. It doesn't find all paths—just the locally best.
       So it might miss valid paths like direct edges if they're not the max at each step.

       Parameters:
           matrix (pd.DataFrame): Pruned symmetric correlation matrix (zero means no edge).
           start_node (str): The starting attribute.
           end_node (str): The target attribute.

       Returns:
           tuple: (path as list of nodes, total correlation sum), or None if path not found.
       """

    # If start and end are the same, there is no meaningful path
    if start_node == end_node:
        return None

    visited = {start_node}
    path = [start_node]
    current = start_node
    total_correlation = 0

    while True:
        # Get unvisited neighbors
        neighbors = matrix.loc[current].drop(labels=visited)

        if neighbors.empty:
            break

        # Pick the neighbor with the highest absolute correlation
        next_node = neighbors.abs().idxmax()
        weight = matrix.loc[current, next_node]

        if weight == 0:
            break

        path.append(next_node)
        visited.add(next_node)
        total_correlation += weight
        current = next_node

        # Stop if we reached the goal
        if current == end_node:
            break
    if path[-1] != end_node:
        print(f"{' → '.join(path)}")
        return None
    else:
        return path, total_correlation


def greedy_dfs_paths(matrix, start_node, end_node):
    """
    Perform a depth-first search (DFS) that always follows the highest absolute correlation neighbors
    at each step (greedy filter). This means if multiple neighbors share the same maximum correlation,
    the function explores all of them, thus supporting multiple greedy paths from start_node to end_node.

    Because this search is greedy, it may not find a path even if one exists —
    if the path requires taking edges with lower correlations at any step, those are skipped.
    As a result, this function only finds paths that consistently follow the strongest connections.

    Parameters:
        matrix (pd.DataFrame): Pruned symmetric correlation matrix.
        start_node (str): Start attribute.
        end_node (str): Target attribute.

    Returns:
        List of (path, total_sum) tuples, or None if no path is found.
    """
    results = []  # List of tuples: (path, total_correlation)
    unfinished_paths = []

    def dfs(node, visited, path, total_sum):
        if node == end_node:
            results.append((path.copy(), total_sum))
            return

        neighbors = matrix.loc[node].drop(labels=visited)
        neighbors = neighbors[neighbors != 0]  # remove zero-weight edges

        if neighbors.empty:
            unfinished_paths.append((path.copy(), total_sum))
            return

        # Greedy step: only keep neighbors with max abs correlation
        max_weight = neighbors.abs().max()
        greedy_neighbors = neighbors[neighbors.abs() == max_weight]

        for neighbor, weight in greedy_neighbors.items():
            visited.add(neighbor)
            path.append(neighbor)
            dfs(neighbor, visited, path, total_sum + weight)
            path.pop()
            visited.remove(neighbor)

    dfs(start_node, {start_node}, [start_node], 0)
    if results:
        return results
    else:
        best_path, best_sum = max(unfinished_paths, key=lambda x: x[1])
        print("Best unfinished path:")
        print(" → ".join(best_path))
        return None


def dfs_paths(matrix, start_node, end_node):
    """
    Explore all possible paths and identify the one with max total correlation.
    """

    all_paths = []

    def dfs(current, visited, path, total_sum):
        print(f"Current path: {path}, total correlation sum: {total_sum}")

        if current == end_node:
            print(f"*** Reached end node! Final path: {path}, total correlation: {total_sum} ***")
            all_paths.append((path.copy(), total_sum))
            return

        neighbors = matrix.loc[current].drop(labels=visited)
        for neighbor, weight in neighbors.items():
            if weight == 0:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            dfs(neighbor, visited, path, total_sum + weight)
            path.pop()
            visited.remove(neighbor)

    dfs(start_node, {start_node}, [start_node], 0)
    return all_paths


def astar_max_correlation(matrix, start, goal, threshold):
    # Priority queue (open list): min-heap simulating max-heap using negative f and g values
    # Each element is a tuple: (-f, -g, current_node, path_so_far)
    #   f = g + h, where:
    #       g = actual total correlation sum so far
    #       h = estimated remaining correlation (heuristic)
    queue = [(0, 0, start, [start])]

    # Continue exploring while there are paths to expand
    while queue:
        # Pop the node with the highest f (total estimated correlation)
        neg_f, neg_g, current, path = heapq.heappop(queue)
        g = -neg_g  # convert back to positive actual correlation sum

        # If we've reached the goal, return the path and the total correlation
        if current == goal:
            return path, g

        # Explore neighbors of the current node
        for neighbor in matrix.columns:
            # Skip if it's the current node or already in the path (prevents cycles)
            if neighbor == current or neighbor in path:
                continue

            # Get correlation value between current and neighbor
            corr = matrix.loc[current, neighbor]

            # Only consider significant correlations
            if abs(corr) > threshold:
                # Extend the current path
                new_path = path + [neighbor]
                # Update actual correlation sum (g)
                new_g = g + corr   # trying to maximize this
                # Estimate potential future correlation (h) using heuristic
                h = heuristic(matrix, neighbor, goal, new_path, threshold)
                # h = heuristic(matrix, neighbor, goal, new_path, threshold, sigma)
                # Total estimated cost (f = g + h)
                f = new_g + h
                # Push the new state into the queue with negated values
                heapq.heappush(queue, (-f, -new_g, neighbor, new_path))

    return None, 0


def heuristic(matrix, current, goal, visited, threshold):
    def dfs(node, visited_set):
        """
        Estimates the maximum achievable correlation from `current` to `goal`
        by performing a depth-first search (DFS). The heuristic returns the highest
        possible sum of correlation coefficients along any acyclic path from
        `current` to `goal`, excluding already visited nodes and weak correlations.

        This is used as the heuristic `h(n)` in A* to guide the search.
        """
        if node == goal:
            return 0  # No cost if already at goal

        # Track the best (maximum) total correlation found from this node
        max_corr_sum = float('-inf')
        for neighbor in matrix.columns:
            # Skip already visited nodes to prevent cycles
            if neighbor in visited_set:
                continue
            # Get the correlation value between `node` and `neighbor`
            corr = matrix.loc[node, neighbor]
            # Ignore edges with correlation below the threshold
            if abs(corr) <= threshold:
                continue
            # Mark the neighbor as visited for this DFS branch
            new_visited = visited_set | {neighbor}
            # Recursively explore paths from the neighbor to the goal
            remaining_corr = dfs(neighbor, new_visited)
            # If a valid path was found, update the maximum total correlation
            if remaining_corr != float('-inf'):
                total_corr = corr + remaining_corr
                max_corr_sum = max(max_corr_sum, total_corr)
        # If no valid paths were found, return 0
        return max_corr_sum if max_corr_sum != float('-inf') else 0

    # Convert the visited path list to a set for fast lookup
    visited_set = set(visited)
    # Start DFS from the current node
    return dfs(current, visited_set)


def run_selected_path_finding_method(method, matrix, start, end):
    if method == 'greedy':
        result = greedy_correlation_path(matrix, start_node=start, end_node=end)
        if result is not None:
            path, score = result
            print(f"Greedy path: {' → '.join(path)}\n")
            return [path], [score]
        else:
            print("No path found using Greedy.\nNote: Greedy is shortsighted. "
                  "It doesn't find all paths — just the locally best.\n"
                  "So it might miss valid paths like direct edges if they're not the max at each step.")
            return None, None

    elif method == 'greedy+dfs':
        results = greedy_dfs_paths(matrix, start_node=start, end_node=end)
        if results:
            print("Greedy+DFS paths:")
            paths, scores = zip(*results)
            for path, score in results:
                print(f"{' → '.join(path)} | Score: {round(score, 2)}")
            print("\n")
            return list(paths), list(scores)
        else:
            print("No path found using Greedy+DFS.\n")
            return None, None

    elif method == 'dfs':
        results = dfs_paths(matrix, start_node=start, end_node=end)
        if not results:
            print("No paths found using DFS.\n")
            return None, None
        else:
            print("All paths (DFS):")
            paths, scores = zip(*results)
            for path, score in results:
                print(f"Path: {' → '.join(path)} | Score: {round(score, 2)}")
            best_index = scores.index(max(scores))
            print(f"Best path: {' → '.join(paths[best_index])} | Score: {round(scores[best_index], 2)}\n")
            return list(paths), list(scores)

    elif method == 'a_star':
        path, score = astar_max_correlation(matrix, start=start, goal=end, threshold=0)
        if path:
            print(f"A* Path: {' → '.join(path)} | Score: {round(score, 2)}\n")
            return [path], [score]
        else:
            print("No path found using A*.\n")
            return None, None
    return None
