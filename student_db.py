import sqlite3
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json

class StudentDB:
    def __init__(self, db_path: str = 'student.db'):
        """Initialize the database connection and create tables if they don't exist."""
        self.db_path = db_path
        self._create_tables()
    
    def _get_connection(self):
        """Create and return a database connection."""
        return sqlite3.connect(self.db_path)
    
    def _create_tables(self):
        """Create the students table if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    zpd_score REAL DEFAULT 5.0,
                    zp_history TEXT DEFAULT '[]',  -- Store as JSON array
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def add_student(self, student_id: str, student_name: str, initial_zpd: float = 5.0) -> bool:
        """
        Add a new student to the database.
        
        Args:
            student_id: Unique identifier for the student
            student_name: Full name of the student
            initial_zpd: Initial ZPD score (default: 5.0)
            
        Returns:
            bool: True if successful, False if student_id already exists
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO students (student_id, student_name, zpd_score, zp_history)
                    VALUES (?, ?, ?, ?)
                ''', (student_id, student_name, initial_zpd, '[5.0]'))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False  # Student ID already exists
    
    def update_zpd_score(self, student_id: str, new_zpd: float) -> bool:
        """
        Update a student's ZPD score and add to history.
        
        Args:
            student_id: ID of the student to update
            new_zpd: New ZPD score
            
        Returns:
            bool: True if update was successful, False if student not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Get current history
            cursor.execute('SELECT zp_history FROM students WHERE student_id = ?', (student_id,))
            result = cursor.fetchone()
            
            if not result:
                return False
                
            # Parse current history
            zp_history = json.loads(result[0])
            
            # Add new score and keep only last 10 entries
            zp_history.append(new_zpd)
            zp_history = zp_history[-10:]
            
            # Update with new score and history
            cursor.execute('''
                UPDATE students 
                SET zpd_score = ?, 
                    zp_history = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE student_id = ?
            ''', (new_zpd, json.dumps(zp_history), student_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_student(self, student_id: str) -> Optional[Dict]:
        """
        Retrieve a student's information.
        
        Args:
            student_id: ID of the student to retrieve
            
        Returns:
            Optional[Dict]: Student information or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT student_id, student_name, zpd_score, zp_history, created_at, last_updated
                FROM students
                WHERE student_id = ?
            ''', (student_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'student_id': row[0],
                    'student_name': row[1],
                    'zpd_score': row[2],
                    'zp_history': json.loads(row[3]),
                    'created_at': row[4],
                    'last_updated': row[5]
                }
            return None
    
    def get_all_students(self) -> List[Dict]:
        """
        Retrieve all students' information.
        
        Returns:
            List[Dict]: List of all students' information
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT student_id, student_name, zpd_score, zp_history, created_at, last_updated
                FROM students
                ORDER BY student_name
            ''')
            
            return [{
                'student_id': row[0],
                'student_name': row[1],
                'zpd_score': row[2],
                'zp_history': json.loads(row[3]),
                'created_at': row[4],
                'last_updated': row[5]
            } for row in cursor.fetchall()]
    
    def get_zpd_history(self, student_id: str) -> list[float]:
        """
        Get the ZPD score history for a student.
        
        Args:
            student_id: ID of the student
            
        Returns:
            List of ZPD scores, most recent last
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT zp_history FROM students WHERE student_id = ?', (student_id,))
            result = cursor.fetchone()
            
            if not result or not result[0]:
                return []
                
            return json.loads(result[0])

# Example usage
if __name__ == "__main__":
    # Initialize the database
    db = StudentDB()
    
    # Add a test student
    db.add_student("S001", "John Doe", 5.0)
    
    # Update ZPD score
    db.update_zpd_score("S001", 5.5)
    
    # Get student info
    student = db.get_student("S001")
    print(f"Student: {student}")
    
    # Get all students
    all_students = db.get_all_students()
    print(f"All students: {all_students}")
    
    # Get ZPD history
    zpd_history = db.get_zpd_history("S001")
    print(f"ZPD history: {zpd_history}")
