from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    score: Optional[float] = None

    def get_point1(self) -> Tuple[float, float]:
        return self.x, self.y

    def get_point2(self) -> Tuple[float, float]:
        return self.x + self.width, self.y + self.height
