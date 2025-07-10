"""
Student Manager Module

Handles student session management and database interactions.
"""
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import time
from datetime import datetime, timedelta
from student_db import StudentDB
from ZPD_calculator import ZPDCalculator

@dataclass
class StudentSession:
    """Represents a student's session data with timeout functionality."""
    student_id: str
    student_name: str
    current_zpd: float
    db: StudentDB
    _zpd_calculator: ZPDCalculator = field(init=False, repr=False)
    _last_activity: float = field(init=False, repr=False)
    SESSION_TIMEOUT_MINUTES: int = 30  # Session expires after 30 minutes of inactivity
    _zpd_history: list = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize the ZPD calculator and set initial activity time."""
        self._zpd_calculator = ZPDCalculator(initial_zpd=self.current_zpd)
        self._last_activity = time.time()
        self._zpd_history = [self.current_zpd]
    
    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self._last_activity = time.time()
    
    def is_expired(self) -> bool:
        """Check if the session has expired due to inactivity.
        
        Returns:
            bool: True if session has expired, False otherwise
        """
        inactive_duration = time.time() - self._last_activity
        return inactive_duration > (self.SESSION_TIMEOUT_MINUTES * 60)
    
    def get_remaining_session_time(self) -> timedelta:
        """Get the remaining time until session expires.
        
        Returns:
            timedelta: Remaining session time
        """
        inactive_duration = time.time() - self._last_activity
        remaining = (self.SESSION_TIMEOUT_MINUTES * 60) - inactive_duration
        return timedelta(seconds=max(0, remaining))
    
    def update_zpd(self, performance_score: float) -> Tuple[float, float]:
        """
        Update the student's ZPD score based on performance.
        
        Args:
            performance_score: Score between 0.0 and 1.0 representing performance
            
        Returns:
            Tuple of (old_zpd, new_zpd)
        """
        if self.is_expired():
            raise SessionExpiredError("Session has expired. Please log in again.")
            
        self.update_activity()  # Update activity on ZPD update
        old_zpd = self.current_zpd
        
        # Update ZPD using the calculator
        new_zpd = self._zpd_calculator.update_user_zpd(performance_score)
        
        # Update in database and session
        self.current_zpd = new_zpd
        self.db.update_zpd_score(self.student_id, new_zpd)
        self._zpd_history.append(new_zpd)
        
        return old_zpd, new_zpd
    
    def get_zpd_history(self) -> list:
        """Get the ZPD history for this session."""
        return self._zpd_history
    
    def get_zpd_trend(self) -> float:
        """Get the ZPD trend for this session."""
        if len(self._zpd_history) < 2:
            return 0.0
        trend = (self._zpd_history[-1] - self._zpd_history[-2]) / (len(self._zpd_history) - 1)
        return trend

class SessionExpiredError(Exception):
    """Exception raised when trying to use an expired session."""
    pass

class StudentManager:
    """Manages student sessions and database interactions with session handling."""
    
    def __init__(self, db_path: str = 'student.db'):
        """Initialize the student manager with a database connection."""
        self.db = StudentDB(db_path)
        self.active_sessions: Dict[str, StudentSession] = {}
    
    def get_session(self, student_id: str) -> Optional[StudentSession]:
        """
        Get an active session for a student if it exists and is not expired.
        
        Args:
            student_id: ID of the student
            
        Returns:
            StudentSession if valid session exists, None otherwise
        """
        session = self.active_sessions.get(student_id)
        if session and not session.is_expired():
            session.update_activity()
            return session
        if session:
            # Clean up expired session
            del self.active_sessions[student_id]
        return None
    
    def create_session(self, student_id: str, student_name: str, initial_zpd: float) -> StudentSession:
        """
        Create a new session for a student.
        
        Args:
            student_id: ID of the student
            student_name: Name of the student
            initial_zpd: Initial ZPD score for the session
            
        Returns:
            New StudentSession instance
        """
        # End any existing session for this user
        if student_id in self.active_sessions:
            del self.active_sessions[student_id]
            
        session = StudentSession(
            student_id=student_id,
            student_name=student_name,
            current_zpd=initial_zpd,
            db=self.db
        )
        self.active_sessions[student_id] = session
        return session
    
    def end_session(self, student_id: str) -> None:
        """End a student's session."""
        if student_id in self.active_sessions:
            del self.active_sessions[student_id]
    
    def update_student_zpd(self, student_session: StudentSession, is_correct: bool, is_partial: bool = False) -> Tuple[float, float]:
        """
        Update a student's ZPD based on their answer.
        
        Args:
            student_session: The student's session
            is_correct: Whether the answer was correct
            is_partial: Whether the answer was partially correct
            
        Returns:
            Tuple of (old_zpd, new_zpd)
            
        Raises:
            SessionExpiredError: If the session has expired
        """
        # This will raise SessionExpiredError if session is invalid
        if student_session.is_expired():
            self.end_session(student_session.student_id)
            raise SessionExpiredError("Session has expired. Please log in again.")
            
        # Calculate performance score (1.0 = correct, 0.5 = partial, 0.0 = incorrect)
        performance_score = 1.0 if is_correct else (0.5 if is_partial else 0.0)
        return student_session.update_zpd(performance_score)
    
    def get_or_create_student(self) -> 'StudentSession':
        """
        Get an existing student or create a new one, and create a session.
        
        Returns:
            StudentSession: The student's session data
        """
        student_id = input("Enter your student ID: ").strip()
        
        # Check for existing valid session first
        existing_session = self.get_session(student_id)
        if existing_session:
            print(f"\nWelcome back, {existing_session.student_name}!")
            print(f"Your current ZPD score: {existing_session.current_zpd:.1f}")
            print(f"Session will expire in {existing_session.get_remaining_session_time()}")
            
            # Show ZPD history
            zpd_history = existing_session.get_zpd_history()
            if len(zpd_history) > 1:  # Only show if we have history
                print("\nYour ZPD history:")
                for i, score in enumerate(zpd_history[-5:], 1):  # Show last 5 scores
                    print(f"  Session {i}: {score:.1f}")
                
                # Show trend
                trend = existing_session.get_zpd_trend()
                if abs(trend) > 0.1:  # Only show significant trends
                    direction = "↑" if trend > 0 else "↓"
                    print(f"\nYour ZPD is trending {direction} by an average of {abs(trend):.1f} per session")
            
            return existing_session
            
        student = self.db.get_student(student_id)
        
        if student:
            print(f"\nWelcome back, {student['student_name']}!")
            print(f"Your current ZPD score: {student['zpd_score']:.1f}")
            
            # Show ZPD history for returning students
            session = self.create_session(
                student_id=student_id,
                student_name=student['student_name'],
                initial_zpd=student['zpd_score']
            )
            
            zpd_history = session.get_zpd_history()
            if len(zpd_history) > 1:  # Only show if we have history
                print("\nYour ZPD history:")
                for i, score in enumerate(zpd_history[-5:], 1):
                    print(f"  Session {i}: {score:.1f}")
            
            return session
        else:
            student_name = input("\nWelcome new student! Please enter your name: ").strip()
            initial_zpd = 5.0  # Default starting ZPD
            self.db.add_student(student_id, student_name, initial_zpd)
            print(f"\nWelcome, {student_name}! Your starting ZPD score is {initial_zpd:.1f}")
            
            return self.create_session(
                student_id=student_id,
                student_name=student_name,
                initial_zpd=initial_zpd
            )
