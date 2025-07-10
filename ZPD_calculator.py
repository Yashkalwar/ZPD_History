"""
ZPD Calculator Module

Handles the calculation of Zone of Proximal Development scores.
"""
from typing import List
import numpy as np

class ZPDCalculator:
    def __init__(self, initial_zpd: float = 5.0):
        """
        Initialize the ZPD calculator with an initial ZPD score.
        
        Args:
            initial_zpd: Initial ZPD score (default: 5.0)
        """
        self.learning_rate = 0.2  # How quickly the ZPD score changes
        self.current_zpd = initial_zpd
        self.min_zpd = 1.0
        self.max_zpd = 10.0

    def get_user_zpd(self) -> float:
        """Get current ZPD score (1.0 to 10.0)"""
        return self.current_zpd

    def calculate_performance_score(self, scores: List[float]) -> float:
        """
        Calculate weighted performance score (0.0 to 1.0)
        
        Args:
            scores: List of performance scores (0.0 to 1.0)
            
        Returns:
            Weighted average performance score
        """
        if not scores:
            return 0.5  # Neutral score if no history
            
        # Weight recent scores more heavily
        weights = np.linspace(0.1, 1.0, len(scores))
        weighted_sum = sum(w * s for w, s in zip(weights, scores))
        return min(1.0, max(0.0, weighted_sum / sum(weights)))

    def update_user_zpd(self, performance_score: float) -> float:
        """
        Update ZPD score based on performance
        
        Args:
            performance_score: Current performance score (0.0 to 1.0)
            
        Returns:
            New ZPD score
        """
        old_zpd = self.current_zpd
        
        # Calculate adjustment based on performance
        # performance_score is between 0.0 and 1.0, so we subtract 0.5 to get -0.5 to 0.5 range
        adjustment = self.learning_rate * (performance_score - 0.5)
        
        # Calculate new ZPD, keeping it within bounds
        self.current_zpd = max(self.min_zpd, min(self.max_zpd, old_zpd + adjustment))
        
        # Debug output
        print(f"[ZPD CALC] Current: {old_zpd:.2f}, "
              f"Performance: {performance_score:.2f}, "
              f"Adjustment: {adjustment:+.2f}, "
              f"New ZPD: {self.current_zpd:.2f}")
        
        return self.current_zpd