from abc import ABC
from typing import List


class Vertex(ABC):
    coords: List[..., float]


class Vertex2D(Vertex):
    def __init__(self, x: float, y: float):
        self.coords = [x, y]


class Vertex3D(Vertex):
    def __init__(self, x: float, y: float, z: float):
        self.coords = [x, y, z]


class Polygon:
    vertices: List[Vertex]

    def __init__(self, vertices: List[Vertex]):
        self.vertices = vertices

    def __len__(self):
        return len(self.vertices)
