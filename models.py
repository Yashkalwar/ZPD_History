# models.py
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    
    # Relationship to ZPD data
    zpd_data = relationship("UserZPD", back_populates="user", uselist=False)

class UserZPD(Base):
    __tablename__ = 'user_zpd'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    zpd_score = Column(Float, default=9.5)  # 1.0 to 10.0 scale
    performance_history = Column(JSON, default=list)  # Stores list of recent scores (0.0 to 1.0)
    
    # Relationship back to User
    user = relationship("User", back_populates="zpd_data")