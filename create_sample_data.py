from student_db import StudentDB
import time

def add_sample_students():
    """Add sample students with their current ZPD scores."""
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            db = StudentDB('student.db')
            connection = db._get_connection()
            
            # Sample students with their current ZPD scores
            students = [
                {'id': 'S1001', 'name': 'Agent A', 'zpd': 2.5},
                {'id': 'S1002', 'name': 'Agent B', 'zpd': 6.0},
                {'id': 'S1003', 'name': 'Agent C', 'zpd': 5.5},
                {'id': 'S1004', 'name': 'Agent D', 'zpd': 9.0}
            ]
            
            for student in students:
                try:
                    if db.get_student(student['id']):
                        print(f"Updating {student['name']} (ID: {student['id']}) - ZPD: {student['zpd']}")
                        connection.execute(
                            'UPDATE students SET student_name = ?, zpd_score = ? WHERE student_id = ?',
                            (student['name'], student['zpd'], student['id'])
                        )
                    else:
                        print(f"Adding new student: {student['name']} (ID: {student['id']}) - ZPD: {student['zpd']}")
                        db.add_student(student['id'], student['name'], student['zpd'])
                    
                    connection.commit()
                    
                except Exception as e:
                    print(f"Error processing student {student['id']}: {e}")
                    connection.rollback()
                    time.sleep(1)  # Wait a bit before retrying
                    continue
            
            # If we get here, all updates were successful
            print("\nCurrent students in database:")
            print("-" * 50)
            for student in db.get_all_students():
                print(f"ID: {student['student_id']} | Name: {student['student_name']} | Current ZPD: {student['zpd_score']:.1f}")
            
            return  # Success, exit the function
            
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("\n❌ Failed to update student data after multiple attempts.")
                print("Please make sure no other instances of the application are running.")
                print("If the problem persists, you may need to delete the student.db file and try again.")
                return

if __name__ == "__main__":
    add_sample_students()