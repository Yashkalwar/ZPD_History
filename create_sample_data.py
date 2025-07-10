from student_db import StudentDB

def add_sample_students():
    """Add sample students with their current ZPD scores."""
    db = StudentDB('student.db')
    
    # Sample students with their current ZPD scores
    students = [
        {'id': 'S1001', 'name': 'Agent A', 'zpd': 4.5},
        {'id': 'S1002', 'name': 'Agent B', 'zpd': 6.0},
        {'id': 'S1003', 'name': 'Agent C', 'zpd': 5.5},
        {'id': 'S1004', 'name': 'Agent D', 'zpd': 7.0}
    ]
    
    for student in students:
        if db.get_student(student['id']):
            print(f"Updating {student['name']} (ID: {student['id']}) - ZPD: {student['zpd']}")
            # Update existing student
            db._get_connection().execute(
                'UPDATE students SET student_name = ?, zpd_score = ? WHERE student_id = ?',
                (student['name'], student['zpd'], student['id'])
            )
        else:
            print(f"Adding new student: {student['name']} (ID: {student['id']}) - ZPD: {student['zpd']}")
            # Add new student
            db.add_student(student['id'], student['name'], student['zpd'])
        
        db._get_connection().commit()
    
    print("\nCurrent students in database:")
    print("-" * 50)
    for student in db.get_all_students():
        print(f"ID: {student['student_id']} | Name: {student['student_name']} | Current ZPD: {student['zpd_score']:.1f}")

if __name__ == "__main__":
    add_sample_students()