from sqlalchemy.orm import Session
from models import UserZPD
from typing import List, Tuple
import numpy as np

class ZPDCalculator:
    def __init__(self, db: Session):
        self.learning_rate = 0.5  # How quickly the ZPD score changes
        self.db = db
        self.min_zpd = 1.0
        self.max_zpd = 10.0

    def get_user_zpd(self, user_id: int) -> float:
        """Get current ZPD score for a user (1.0 to 10.0)"""
        user_zpd = self.db.query(UserZPD).filter(UserZPD.user_id == user_id).first()
        
        if not user_zpd:
            # Start new users at 5.5 (middle of the scale)
            user_zpd = UserZPD(
                user_id=user_id,
                zpd_score=5.5,
                performance_history=[]
            )
            self.db.add(user_zpd)
            self.db.commit()
            self.db.refresh(user_zpd)
        
        return user_zpd.zpd_score

    def calculate_performance_score(self, scores: List[float]) -> float:
        """Calculate weighted performance score (0.0 to 1.0)"""
        if not scores:
            return 0.5  # Neutral score if no history
            
        # Weight recent scores more heavily
        weights = np.linspace(0.1, 1.0, len(scores))
        weighted_sum = sum(w * s for w, s in zip(weights, scores))
        return min(1.0, max(0.0, weighted_sum / sum(weights)))

    def update_user_zpd(self, user_id: int, recent_scores: List[float]) -> Tuple[float, str]:
        """
        Update ZPD score based on recent performance
        
        Args:
            user_id: ID of the user
            recent_scores: List of recent scores (0.0 to 1.0)
            
        Returns:
            Tuple of (new_zpd_score, message)
        """
        user_zpd = self.db.query(UserZPD).filter(UserZPD.user_id == user_id).first()
        
        # Create new user if doesn't exist
        if not user_zpd:
            user_zpd = UserZPD(
                user_id=user_id,
                zpd_score=5.5,  # Start in the middle
                performance_history=[]
            )
            self.db.add(user_zpd)
        
        # Update performance history (keeping last 100 scores)
        user_zpd.performance_history = (user_zpd.performance_history or []) + recent_scores
        user_zpd.performance_history = user_zpd.performance_history[-100:]
        
        # Calculate performance (0.0 to 1.0)
        performance = self.calculate_performance_score(recent_scores)
        
        # Calculate adjustment (positive or negative based on performance)
        # 0.5 is neutral, below decreases ZPD, above increases it
        adjustment = self.learning_rate * (performance - 0.5)
        
        # Apply adjustment, keeping within 1.0-10.0 range
        new_zpd = max(self.min_zpd, min(self.max_zpd, user_zpd.zpd_score + adjustment))
        
        # Update the score
        user_zpd.zpd_score = new_zpd
        
        # Save changes
        self.db.commit()
        self.db.refresh(user_zpd)
        
        # Generate appropriate message
        if adjustment > 0.1:
            message = f"Great job! ZPD increased to {new_zpd:.1f}"
        elif adjustment < -0.1:
            message = f"ZPD adjusted to {new_zpd:.1f}. Keep practicing!"
        else:
            message = f"ZPD is now {new_zpd:.1f}"
        
        return new_zpd, message