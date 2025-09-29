#To evaluate the performance of the AUV in every tasks
import os
import sys
import math
from typing import Tuple
class DistanceCalculator:
    """Calculate distance between robot current coordinates and target objects"""
    
    def __init__(self):
        # Target object coordinates data
        self.target_coordinates = {
            "MINING ROBOT": [
                (-71, 149, -61),
                (325, -47, -83)
            ],
            "OIL PIPELINE": [
                (345, -165, -32),
                (539, -233, -42),
                (207, -30, -66)
            ],
            "OIL DRUM": [
                (447, -203, -98)
            ],
            "SUNKEN SHIP": [
                (429, -151, -69),
                (78, -11, -47)

            ],
            "ELECTRICAL BOX": [
                (168, 168, -65)
            ],
            "WIND POWER STATION": [
                (207, -30, -66)
            ],
            "AIRCRAFT WRECKAGE": [
                (40, -9, -54),
                (296, 78, -70),
                (292, -186, -67)
            ],
            "H-MARKED LANDING PLATFORM": [
                (267, 33, -80)
            ]
        }

    def calculate_3d_distance(self, point1: Tuple[float, float, float], 
                             point2: Tuple[float, float, float]) -> float:
        """
        Calculate Euclidean distance between two 3D coordinate points
        
        Args:
            point1: First point coordinates (x1, y1, z1)
            point2: Second point coordinates (x2, y2, z2)
            
        Returns:
            float: Distance between the two points
        """
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return distance

    def calculate_2d_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two 2D coordinate points

        Args:
            point1: First point coordinates (x1, y1)
            point2: Second point coordinates (x2, y2)

        Returns:
            float: Distance between the two points
        """
        x1, y1 = point1
        x2, y2 = point2

        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance