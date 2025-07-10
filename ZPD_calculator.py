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
        self.current_zpd = round(initial_zpd, 1)
        self.min_zpd = 1.0
        self.max_zpd = 10.0
        
        # Smoothing factors (0.0 to 1.0)
        self.performance_alpha = 0.3  # How quickly to update performance estimate
        self.zpd_beta = 0.15         # How quickly to adjust ZPD
        
        # Track performance trend
        self.smoothed_performance = 0.5
        self.performance_trend = 0.0

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
        Update ZPD score using exponential moving averages for smooth updates.
        
        Args:
            performance_score: Current performance score (0.0 to 1.0)
            
        Returns:
            New ZPD score (rounded to 1 decimal place)
        """
        old_zpd = self.current_zpd
        
        # Update smoothed performance (EMA)
        prev_smoothed = self.smoothed_performance
        self.smoothed_performance = (
            self.performance_alpha * performance_score + 
            (1 - self.performance_alpha) * prev_smoothed
        )
        
        # Calculate performance trend (derivative of smoothed performance)
        self.performance_trend = self.smoothed_performance - prev_smoothed
        
        # Calculate target ZPD adjustment based on performance
        if performance_score >= 0.9:  # Very good answer
            # Count consecutive successes
            if not hasattr(self, 'consecutive_successes'):
                self.consecutive_successes = 0
            self.consecutive_successes += 1
            
            # Base adjustment + bonus for streak (capped at 3)
            streak_bonus = min(3, self.consecutive_successes) * 0.15
            adjustment = 0.25 + streak_bonus
            
        elif performance_score >= 0.6:  # Partially correct answer
            # Reset success streak on partial answers
            if hasattr(self, 'consecutive_successes'):
                self.consecutive_successes = 0
                
            # Small positive adjustment for partial correctness
            # Scale adjustment based on how close to 1.0 the score is
            partial_bonus = (performance_score - 0.6) * 0.4
            adjustment = 0.1 + partial_bonus
            
        else:  # Incorrect answer (performance_score < 0.6)
            # Reset success streak on incorrect answers
            if hasattr(self, 'consecutive_successes'):
                self.consecutive_successes = 0
                
            # Scale penalty based on how wrong the answer was
            # But cap the maximum penalty to prevent large drops
            penalty = (0.6 - performance_score) * 0.3
            adjustment = -min(0.15, penalty)  # Cap penalty at -0.15
        
        # Apply non-linear scaling to prevent large jumps
        adjustment = np.sign(adjustment) * (abs(adjustment) ** 0.5) * 0.5
        
        # Update ZPD using EMA for smooth transitions
        target_zpd = old_zpd + adjustment
        self.current_zpd = round(
            max(self.min_zpd, min(self.max_zpd, 
                self.zpd_beta * target_zpd + 
                (1 - self.zpd_beta) * old_zpd
            )), 
            1  # Round to 1 decimal place
        )
        
        # Debug output
        print(f"[ZPD EMA] Current: {old_zpd:.1f}, "
              f"Performance: {performance_score:.2f} (smooth: {self.smoothed_performance:.2f}), "
              f"Trend: {self.performance_trend:+.3f}, "
              f"Adjustment: {adjustment:+.3f}, "
              f"New ZPD: {self.current_zpd:.1f}")
        
        return self.current_zpd