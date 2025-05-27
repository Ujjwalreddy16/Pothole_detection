import heapq

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position  # (x, y)
        self.parent = parent
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic cost to goal
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f


def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_inside_bbox(point, bbox):
    """Check if a point is inside a bounding box (x, y, w, h)"""
    x, y = point
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def a_star(grid_size, start, goal, pothole_bboxes):
    """Performs A* pathfinding on a grid while avoiding pothole bounding boxes."""
    open_list = []
    closed_set = set()
    start_node = Node(start, None, 0, heuristic(start, goal))
    heapq.heappush(open_list, start_node)

    max_x, max_y = grid_size

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.position == goal:
            # Reconstruct path
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Reverse path

        closed_set.add(current_node.position)

        x, y = current_node.position
        neighbors = [(x+dx, y+dy) for dx, dy in
                     [(-1,0), (1,0), (0,-1), (0,1)]]  # 4-directional movement

        for neighbor_pos in neighbors:
            nx, ny = neighbor_pos
            if (0 <= nx < max_x) and (0 <= ny < max_y) and neighbor_pos not in closed_set:
                # Avoid pothole areas
                if any(is_inside_bbox(neighbor_pos, bbox) for bbox in pothole_bboxes):
                    continue

                g_cost = current_node.g + 1
                h_cost = heuristic(neighbor_pos, goal)
                neighbor_node = Node(neighbor_pos, current_node, g_cost, h_cost)

                # Add to open list if not already explored
                heapq.heappush(open_list, neighbor_node)

    return None  # No path found
