import copy
from math import sqrt, hypot

import pygame as pg
import pygame_menu as pgm

import time

from abc import ABC, abstractmethod

# Global configuration
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Colors
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
LIGHTYELLOW = (255, 255, 51)
BLACK = (0, 0, 0)
ORANGE = (255, 128, 0)

EPS = 0.000000001  # used to round triangle surface calculation


#####################

class ZoneEqArray:
    def __init__(self):
        self.zoneIndex = [0]  # zone index array
        self.nextZoneNumber = 0  # last used zone number

    # precondition len(self.zoneIndex) >= 1
    # postcondition each zone index is minimal
    def _overall_zone_index_update(self):
        for i in range(1, len(self.zoneIndex)):
            if i > self.zoneIndex[i] > self.zoneIndex[self.zoneIndex[i]]:
                self.zoneIndex[i] = self.zoneIndex[self.zoneIndex[i]]

    # erzeugt den nächsten Zonenindex
    def make_zone(self):
        self.nextZoneNumber += 1
        self.zoneIndex.append(self.nextZoneNumber)
        return self.nextZoneNumber

    def merge_zones(self, idx1, idx2):
        while self.zoneIndex[idx1] != idx1:
            idx1 = self.zoneIndex[idx1]

        while self.zoneIndex[idx2] != idx2:
            idx2 = self.zoneIndex[idx2]

        if idx1 < idx2:
            self.zoneIndex[idx2] = idx1
        elif idx1 > idx2:
            self.zoneIndex[idx1] = idx2

            # self._overallZoneIndexUpdate()

    def get_zone_id(self, idx):
        while self.zoneIndex[idx] != idx:
            idx = self.zoneIndex[idx]
        return idx


class HexBoard:
    def __init__(self, size):
        self.board = None
        self.myZoneEqArray = ZoneEqArray()
        self.redsTurn = True
        self.size = size

        self.neighbours = [(0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1)]
        self._initialize_board(size)

    # noinspection PyTypeChecker
    def _initialize_board(self, n):
        self.board = [[None for _ in range(0, n + 2)] for _ in range(0, n + 2)]
        self.board[0][0] = ('X', 0)
        self.board[0][n + 1] = ('X', 0)
        self.board[n + 1][0] = ('X', 0)
        self.board[n + 1][n + 1] = ('X', 0)

        z1 = self.myZoneEqArray.make_zone()
        z2 = self.myZoneEqArray.make_zone()
        z3 = self.myZoneEqArray.make_zone()
        z4 = self.myZoneEqArray.make_zone()

        for j in range(1, n + 1):
            self.board[0][j] = ('R', z1)
            self.board[n + 1][j] = ('R', z2)

        for i in range(1, n + 1):
            self.board[i][0] = ('B', z3)
            self.board[i][n + 1] = ('B', z4)

    def make_move(self, move):
        if move is not None:
            i = move[0]
            j = move[1]

            if not self.is_terminated() and 1 <= i <= self.size and 0 <= j <= self.size:
                if self.board[i][j] is None:
                    if self.redsTurn:
                        self.board[i][j] = ('R', -1)
                    else:
                        self.board[i][j] = ('B', -1)

                    # set the field and merge the zone indices of neighbours with the same color
                    self._merge_with_neighbour_zones(i, j)

                    # change bitvalue if the game is not terminated
                    if self.redsTurn:
                        self.redsTurn = False
                    else:
                        self.redsTurn = True
                    return True
        return False

    def _merge_with_neighbour_zones(self, i, j):
        for k in range(0, len(self.neighbours)):
            x = i + self.neighbours[k][0]
            y = j + self.neighbours[k][1]

            if not self.board[x][y] is None:
                if self.board[i][j][0] == self.board[x][y][0]:
                    if self.board[i][j][1] == -1:
                        self.board[i][j] = (self.board[i][j][0], self.board[x][y][1])
                    else:
                        if not self.board[i][j][1] == self.board[x][y][1]:
                            self.myZoneEqArray.merge_zones(self.board[i][j][1], self.board[x][y][1])

        if self.board[i][j][1] == -1:
            zone = self.myZoneEqArray.make_zone()
            self.board[i][j] = (self.board[i][j][0], zone)

    def is_terminated(self):
        if self.is_red_winner() or self.is_blue_winner():
            return True
        else:
            return False

    def is_red_winner(self):
        if self.myZoneEqArray.get_zone_id(1) == self.myZoneEqArray.get_zone_id(2):
            return True
        else:
            return False

    def is_blue_winner(self):
        if self.myZoneEqArray.get_zone_id(3) == self.myZoneEqArray.get_zone_id(4):
            return True
        else:
            return False

    def get_possible_moves(self):
        possible_moves = []
        for r in range(1, self.size + 1):
            for c in range(1, self.size + 1):
                if self.board[r][c] is None:
                    possible_moves.append((r, c))
        return possible_moves

    def display_board(self):
        s = ""
        for i in range(0, self.size + 2):
            s += "\n"
            for j in range(0, self.size + 2):
                if self.board[i][j] is None:
                    s += " . "
                else:
                    s += " " + self.board[i][j][0] + " "
        s += "\n"

        # for i in range(0, self.size + 2):
        #     s += "\n"
        #     for j in range(0, self.size + 2):
        #         if self.board[i][j] is None:
        #             s += " . "
        #         else:
        #             s += " " + str(self.myZoneEqArray.get_zone_id(self.board[i][j][1])) + " "
        # s += "\n"

        return s


#############
# Evaluator #
#############

class GraphNode:
    def __init__(self, position, evaluator, value=None):
        self.position = position
        self.value = value
        self.path_length_from_start = float('inf')
        self.path_vertices_from_start = []
        self.evaluator = evaluator
        self.neighbors = None

    def clear_vertex_cache(self):
        self.path_length_from_start = float('inf')
        self.path_vertices_from_start = []

    # def set_owner(self, owner):
    #     self.owner = owner

    def __str__(self):
        # return f"{self.position}, {self.owner}"
        return f"{self.position}"

    def set_neighbors(self):
        self.neighbors = self.evaluator.get_neighbors_for_node(self)


class Evaluator:
    def __init__(self, size):
        self.size = size
        self.nodes = {}
        self.evaluated_scores = {}

    def initialize_hexes(self, board):

        self.nodes = {}
        for row in range(1, self.size + 1):
            for col in range(1, self.size + 1):
                index = f"{row}{col}"
                if board[row][col] is not None:
                    self.nodes[index] = GraphNode(f"{row}{col}", self, board[row][col][0])
                else:
                    self.nodes[index] = GraphNode(f"{row}{col}", self)

        self.nodes["outside_left"] = (GraphNode("outside_left", self, "B"))
        self.nodes["outside_right"] = (GraphNode("outside_right", self, "B"))
        self.nodes["outside_top"] = (GraphNode("outside_top", self, "R"))
        self.nodes["outside_bottom"] = (GraphNode("outside_bottom", self, "R"))

        for hex in self.nodes.values():
            hex.set_neighbors()

    def get_neighbors_for_node(self, node):
        if node == self.nodes["outside_left"]:
            return [node for node in self.nodes.values() if node.position[1] == "1"]
        if node == self.nodes["outside_right"]:
            return [node for node in self.nodes.values() if node.position[1] == f"{self.size}"]
        if node == self.nodes["outside_top"]:
            return [node for node in self.nodes.values() if node.position[0] == "1"]
        if node == self.nodes["outside_bottom"]:
            return [node for node in self.nodes.values() if node.position[0] == f"{self.size}"]

        row, col = int(node.position[0]), int(node.position[1])
        neighbors = []

        if col > 1:
            # Add left neighbor
            neighbors.append(self.nodes.get(f"{row}{col - 1}"))
        if col < self.size:
            # Add right neighbor
            neighbors.append(self.nodes.get(f"{row}{col + 1}"))
        if row > 1:
            # Add top neighbor
            neighbors.append(self.nodes.get(f"{row - 1}{col}"))
        if row < self.size:
            # Add bottom neighbor
            neighbors.append(self.nodes.get(f"{row + 1}{col}"))
        if row > 1 and col < self.size:
            # Add top-right neighbor
            neighbors.append(self.nodes.get(f"{row - 1}{col + 1}"))
        if row < self.size and col > 1:
            # Add bottom-left neighbor
            neighbors.append(self.nodes.get(f"{row + 1}{col - 1}"))

        if col == 1:
            neighbors.append(self.nodes.get("outside_left"))
        if col == self.size:
            neighbors.append(self.nodes.get("outside_right"))
        if row == self.size:
            neighbors.append(self.nodes.get("outside_bottom"))
        if row == 1:
            neighbors.append(self.nodes.get("outside_top"))

        return neighbors

    def get_shortest_path_from(self, node, destination_node, perspective):
        starting_vertex = node
        if starting_vertex.value == perspective:
            self.find_shortest_paths_using_dijkstra(starting_vertex, perspective)
            path = (starting_vertex, destination_node, destination_node.path_vertices_from_start,
                    destination_node.path_length_from_start)
            return path
        else:
            print("Not perspective")
        return None

    def find_shortest_paths_using_dijkstra(self, start_vertex, perspective):
        # clear every node cache
        for vertex in self.nodes.values():
            vertex.clear_vertex_cache()

        current_vertices = set(self.nodes.values())
        start_vertex.path_length_from_start = 0
        start_vertex.path_vertices_from_start.append(start_vertex)
        current_vertex = start_vertex
        while current_vertex is not None:
            current_vertices.remove(current_vertex)
            not_checked_neighbors = [neighbor for neighbor in current_vertex.neighbors if neighbor in current_vertices]
            filtered_neighbors = [neighbor for neighbor in not_checked_neighbors if
                                  neighbor.value is None or neighbor.value == perspective]
            for neighbor_vertex in filtered_neighbors:
                weight = 0.0 if neighbor_vertex.value == perspective else 1.0
                theoretic_new_weight = current_vertex.path_length_from_start + weight
                if theoretic_new_weight < neighbor_vertex.path_length_from_start:
                    neighbor_vertex.path_length_from_start = theoretic_new_weight
                    neighbor_vertex.path_vertices_from_start = list(current_vertex.path_vertices_from_start)
                    neighbor_vertex.path_vertices_from_start.append(neighbor_vertex)
            if not current_vertices:
                return
            else:
                current_vertex = min(current_vertices, key=lambda vertex: vertex.path_length_from_start)

    def board_to_string(self, board, perspective):
        string_board = f"{perspective}\\"
        for i in range(1, self.size + 1):
            for j in range(1, self.size + 1):
                item = board[i][j]
                if item is None:
                    string_board += "0"
                else:
                    string_board += item[0]
            string_board += "|"
        string_board = string_board.rstrip("|")  # Remove the trailing "|" character
        return string_board

    def get_heuristic_score(self, perspective, board):
        board_string = self.board_to_string(board, perspective)

        if board_string in self.evaluated_scores:
            return self.evaluated_scores[board_string]

        else:
            self.initialize_hexes(board)
            if perspective == "R":
                computer_path = self.get_shortest_path_from(self.nodes.get("outside_top"),
                                                            self.nodes.get("outside_bottom"),
                                                            perspective)
                player_path = self.get_shortest_path_from(self.nodes.get("outside_left"),
                                                          self.nodes.get("outside_right"),
                                                          "B")
            else:
                player_path = self.get_shortest_path_from(self.nodes.get("outside_top"),
                                                          self.nodes.get("outside_bottom"),
                                                          "R")
                computer_path = self.get_shortest_path_from(self.nodes.get("outside_left"),
                                                            self.nodes.get("outside_right"),
                                                            perspective)

            if computer_path[3] == 0:
                computer_score = -100
            else:
                computer_score = computer_path[3]

            # player_path = self.get_shortest_path_from(self.nodes.get("outside_left"), self.nodes.get("outside_right"), "B")
            if player_path[3] == 0:
                player_score = -100
            else:
                player_score = player_path[3]

            # append board value to evaluator
            self.evaluated_scores[board_string] = player_score - computer_score

            return player_score - computer_score


#####################
# Available Players #
#####################

class MachinePlayer(ABC):
    @abstractmethod
    def get_next_move(self, board):
        pass


class HumanPlayer:
    def __init__(self, color):
        self.evaluator = None
        self.color = color

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator


class Minimax(MachinePlayer):
    def __init__(self, color, depth):
        self.color = color
        self.depth = depth
        self.evaluator = None

    # e_x function to determine if board is terminated
    def terminal_function(self, board):
        if (board.is_red_winner() and self.color == "R") or (not board.is_red_winner() and self.color == "B"):
            return (None, None), float("inf")
        else:
            return (None, None), float("-inf")

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def recursion(self, board: HexBoard, max_node, depth):
        v_x = None
        if board.is_terminated():
            v_x = self.terminal_function(board)
        else:
            successors = board.get_possible_moves()
            successor_values = []

            for successor in successors:
                new_board = copy.deepcopy(board)
                new_board.make_move(successor)
                if depth > 0:
                    v_x = self.recursion(new_board, not max_node, depth - 1)
                else:
                    v_x = ((None, None), self.evaluator.get_heuristic_score(self.color, new_board.board))
                successor_values.append((successor, v_x[1]))

            if max_node:
                best = ((None, None), float("-inf"))
                for i in successor_values:
                    if i[1] > best[1] or i[1] == best[1] and best[0] == (None, None):
                        best = i
                return best

            if not max_node:
                best = ((None, None), float("inf"))
                for i in successor_values:
                    if i[1] < best[1] or i[1] == best[1] and best[0] == (None, None):
                        best = i
                return best

        return v_x

    def get_next_move(self, board):
        return self.recursion(board, True, self.depth)[0]


class AlphaBeta(MachinePlayer):
    def __init__(self, color, depth):
        self.depth = depth
        self.color = color
        self.evaluator = None

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def recursion(self, board, max_player, depth, alpha, beta):
        alpha = alpha
        beta = beta

        if board.is_terminated():
            if (board.is_red_winner() and self.color == "R") or (not board.is_red_winner() and self.color == "B"):
                return (None, None), float("inf")
            else:
                return (None, None), float("-inf")

        if depth == 0:
            return (None, None), self.evaluator.get_heuristic_score(self.color, board.board)

        possible_moves = board.get_possible_moves()

        successor_values = []

        # Check if moves are available
        if possible_moves:
            if max_player:
                best_value = ((None, None), float("-inf"))
                for move in possible_moves:
                    # creating new board for each move
                    updated_board = copy.deepcopy(board)
                    updated_board.make_move(move)

                    # Update best move/score
                    temporary_value = self.recursion(updated_board, False, depth - 1, alpha, beta)
                    if best_value[1] < temporary_value[1] or best_value[0] == (None, None):
                        best_value = (move, temporary_value[1])

                    alpha = max(alpha, best_value[1])

                    successor_values.append(best_value)

                    # Beta cutoff
                    if beta < alpha:
                        break


                return best_value

            # Minimizing player
            else:
                best_value = ((None, None), float("inf"))
                for move in possible_moves:
                    # creating new board for each move
                    updated_board = copy.deepcopy(board)
                    updated_board.make_move(move)

                    temporary_value = self.recursion(updated_board, True, depth - 1, alpha, beta)
                    if best_value[1] > temporary_value[1] or best_value[0] == (None, None):
                        best_value = move, temporary_value[1]

                    beta = min(beta, best_value[1])

                    successor_values.append(best_value)

                    # Alpha cutoff
                    if beta < alpha:
                        break

                return best_value
        else:
            return (None, None), self.evaluator.get_heuristic_score(self.color, board.board)

    def get_next_move(self, board):
        best_move = self.recursion(board, True, self.depth, float("-inf"), float("inf"))
        return best_move[0]


########
# View #
########

# Point class is used for coordinates in the gui and can calculate distances between two points
class Point:
    def __init__(self, pos):
        # if len(pos) == 1:
        #     self.x, self.y = pos[0]
        #     self.X, self.Y = list(map(int, pos[0]))
        # else:
        self.x, self.y = pos
        # self.X, self.Y = list(map(int, pos))

    def dist(self, other):
        return hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))

    def __str__(self):
        return '[x:{x}, y:{y}]'.format(x=self.x, y=self.y)

    def __iter__(self):
        """for unpacking"""
        return (x for x in (self.x, self.y))


# Tile is a helper class for the view which holds all information for each Tile to draw the tile
class Tile:
    def __init__(self, origin, row, col, tile_width, tile_height):
        self.row = row
        self.col = col
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.x = origin.x + (self.col + self.row / 2) * self.tile_width
        self.y = origin.y + 0.75 * self.row * self.tile_height

        self.points = [
            (self.x + self.tile_width / 2, self.y),
            (self.x + self.tile_width, self.y + self.tile_height / 4),
            (self.x + self.tile_width, self.y + 3 * self.tile_height / 4),
            (self.x + self.tile_width / 2, self.y + self.tile_height),
            (self.x, self.y + 3 * self.tile_height / 4),
            (self.x, self.y + self.tile_height / 4)
        ]

    def get_points(self):
        return self.points

    def point_in_tile(self, pos):
        point = Point(pos)
        tile_points = list(map(Point, self.points))
        sum_triangle_s = 0
        # -1 so the hole hexagon is used and the last triangle is not missed
        for i in range(-1, 5):
            sum_triangle_s += triangle_surface(Point(tile_points[i]), Point(tile_points[i + 1]), point)
        area = (3 * ((self.tile_height / 2) ** 2) * sqrt(3)) / 2
        return abs(area - sum_triangle_s) < EPS


class HexGui:
    def __init__(self, game_manager):
        pg.init()
        self.game_manager = game_manager

        # creating a screen and give it a Title
        self.screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pg.display.set_caption("Hex Game")

        self.menu = pgm.Menu('Hex-Board Game', WINDOW_WIDTH, WINDOW_HEIGHT, theme=pgm.themes.THEME_BLUE)
        self.menu_mainloop = None

        # Set starting position for the Hex grid
        self.origin = Point((50, 50))
        self.board_size = None
        self.tile_width = None
        self.tile_height = None
        self.tiles = []

        self.clock = pg.time.Clock()

    def show_menu(self):
        # menu = pgm.Menu('Hex-Board Game', WINDOW_WIDTH, WINDOW_HEIGHT, theme=pgm.themes.THEME_BLUE)
        self.menu.add.dropselect("Board size :", [(f"{i}", i) for i in range(1, 11)], default=4,
                                 onchange=self.game_manager.change_board_size)

        items = [("Human Player", "human_player", 0), ("Minimax 1", "minimax", 1), ("AlphaBeta 1", "alphabeta", 1),
                 ("Minimax 2", "minimax", 2), ("AlphaBeta 2", "alphabeta", 2), ("Minimax 3", "minimax", 3), ("AlphaBeta 3", "alphabeta", 3),
                 ("Minimax 4", "minimax", 4), ("AlphaBeta 4", "alphabeta", 4)]

        self.menu.add.dropselect("Player red", items, default=0, onchange=self.game_manager.change_player_red)
        self.menu.add.dropselect("Player blue", items, default=0, onchange=self.game_manager.change_player_blue)
        self.menu.add.button('Play', self.game_manager.initialize_game)
        self.menu.mainloop(self.screen)

    def initialize_board_view(self, size):
        self.menu.disable()
        # Clear screen
        self.screen.fill(WHITE)
        self.board_size = size
        self.tile_width = ((WINDOW_WIDTH - 100) / (self.board_size + (self.board_size - 1) * 0.5))
        self.tile_height = 2 * self.tile_width / sqrt(3)
        self.tiles = [[Tile(self.origin, c, r, self.tile_width, self.tile_height) for r in range(self.board_size)]
                      for c in range(self.board_size)]

    def draw_board(self, board, red_score=0, blue_score=0):
        # destination sides (player colors left right top and bottom)
        # Left and right + 0.5 * tile size
        corner_a = (self.origin.x - 0.5 * self.tile_width, self.origin.y)
        corner_b = (self.origin.x + self.tile_width * self.board_size - 1 / 8 * self.tile_width, self.origin.y)
        corner_c = (self.origin.x + self.tile_width * (self.board_size + (self.board_size - 1) * 0.5 + 0.5),
                    self.origin.y + self.tile_height * (self.board_size + 1) * 3 / 4 - 1 / 2 * self.tile_height)
        corner_d = (self.origin.x + ((self.board_size - 1) * 0.5) * self.tile_width + self.tile_width * 0.2,
                    self.origin.y + self.tile_height * (self.board_size + 1) * 3 / 4 - 1 / 2 * self.tile_height)
        middle_x = (self.origin.x + self.tile_width * (self.board_size + (self.board_size - 1) * 0.5)) / 2
        middle_y = (self.origin.y + self.tile_height * (self.board_size + 1) * 3 / 4 - 1 / 2 * self.tile_height) / 2
        midpoint = (middle_x, middle_y)

        pg.draw.polygon(self.screen, RED, [corner_a, corner_b, midpoint])  # Color red top row
        pg.draw.polygon(self.screen, BLUE, [corner_b, corner_c, midpoint])  # Color Blue right
        pg.draw.polygon(self.screen, RED, [corner_c, corner_d, midpoint])  # Color red bottom row
        pg.draw.polygon(self.screen, BLUE, [corner_d, corner_a, midpoint])  # Color Blue left

        for r in range(self.board_size):
            for c in range(self.board_size):
                if not board[r + 1][c + 1]:
                    self.draw_tile(self.tiles[r][c], WHITE, LIGHTYELLOW)
                elif board[r + 1][c + 1][0] == "R":
                    self.draw_tile(self.tiles[r][c], RED, LIGHTYELLOW)
                elif board[r + 1][c + 1][0] == "B":
                    self.draw_tile(self.tiles[r][c], BLUE, LIGHTYELLOW)
                else:
                    self.draw_tile(self.tiles[r][c], BLACK, LIGHTYELLOW)

        self.display_text(self.screen, f'Red score: {red_score}', 30, RED, (WINDOW_WIDTH * 9 / 10, WINDOW_HEIGHT / 20))
        self.display_text(self.screen, f'Blue score: {blue_score}', 30, BLUE, (WINDOW_WIDTH * 9 / 10, WINDOW_HEIGHT * 2 / 20))
        pg.display.flip()

    def draw_tile(self, tile, fill_color, border_color):
        pg.draw.polygon(self.screen, fill_color, tile.points)
        pg.draw.polygon(self.screen, border_color, tile.points, 2)

    # method to get move from mouse position
    def get_move(self, position):
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.tiles[r][c].point_in_tile(position):
                    return r + 1, c + 1

    def display_text(self, surface, data, size, col, pos):
        txt = str(data)
        font = pg.font.Font(None, size)
        text = font.render(txt, False, BLACK, col)
        rect = text.get_rect(center=pos)
        surface.blit(text, rect)

    def show_scree_game_over(self, board, winner):
        self.display_text(self.screen, 'GAME OVER', 80, ORANGE, (WINDOW_WIDTH / 2, WINDOW_HEIGHT / 3))
        if winner == "Red":
            self.display_text(self.screen, 'Red won the game', 60, RED, (WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2))
        elif winner == "Blue":
            self.display_text(self.screen, 'Blue won the game', 60, BLUE, (WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2))
        pg.display.flip()


########

##############################
# Static functions
##############################
def create_player(algorithm, depth, color, evaluator):
    if algorithm == "minimax":
        machine_player = Minimax(color, depth)
    elif algorithm == "alphabeta":
        machine_player = AlphaBeta(color, depth)
    else:
        machine_player = HumanPlayer(color)

    return machine_player


def triangle_surface(corner_a, corner_b, corner_c):
    """retrun the surface of a triangle"""
    # Flächenformel (Herons Formel)

    a = corner_c.dist(corner_b)
    b = corner_a.dist(corner_c)
    c = corner_a.dist(corner_b)
    p = (a + b + c) / 2
    return sqrt(p * (p - a) * (p - b) * (p - c))


##############################

class GameManager:
    def __init__(self):
        self.started = False
        self.board = None
        self.evaluator = None
        self.view = HexGui(self)
        self.player_red = HumanPlayer("R")
        self.player_blue = HumanPlayer("B")
        self.clock = pg.time.Clock()
        self.board_size = 5

    def run(self):
        if self.started:
            self.start_game()
        else:
            self.view.show_menu()

    def start_game(self):
        self.view.initialize_board_view(self.board_size)
        pg.init()
        self.view.draw_board(self.board.board)
        run = True
        while run:
            self.clock.tick(30)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False

                if event.type == pg.MOUSEBUTTONDOWN:
                    if isinstance(self.get_moving_player(self.board.redsTurn), HumanPlayer):
                        move = self.view.get_move(pg.mouse.get_pos())
                        if self.board.make_move(move):
                            self.view.draw_board(self.board.board, self.player_red.evaluator.get_heuristic_score("R", self.board.board), self.player_blue.evaluator.get_heuristic_score("B", self.board.board))

                # code to configure minimax algorithm
                if isinstance(self.get_moving_player(self.board.redsTurn), MachinePlayer):
                    if self.board.make_move(self.get_moving_player(self.board.redsTurn).get_next_move(self.board)):
                        self.view.draw_board(self.board.board, self.player_red.evaluator.get_heuristic_score("R", self.board.board), self.player_blue.evaluator.get_heuristic_score("B", self.board.board))
                    break

            # Check if game is terminated and show winner screen
            if self.board.is_terminated():
                if self.board.is_red_winner():
                    self.view.show_scree_game_over(self.board.board, "Red")
                else:
                    self.view.show_scree_game_over(self.board.board, "Blue")

        pg.quit()

    def get_moving_player(self, red_turn):
        return self.player_red if red_turn else self.player_blue

    def initialize_game(self):
        self.board = HexBoard(self.board_size)
        self.evaluator = Evaluator(self.board.size)
        self.player_red.set_evaluator(self.evaluator)
        self.player_blue.set_evaluator(self.evaluator)
        self.started = True
        self.run()

    def change_player_red(self, _, algorithm, depth):
        self.player_red = create_player(algorithm, depth, "R", self.evaluator)

    def change_player_blue(self, _, algorithm, depth):
        self.player_blue = create_player(algorithm, depth, "B", self.evaluator)

    def change_board_size(self, _, size):
        self.board_size = size


def main():
    gm = GameManager()
    gm.run()


if __name__ == '__main__':
    main()
