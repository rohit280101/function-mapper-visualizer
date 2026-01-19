
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import pandas as pd
import os

Base = declarative_base()

class TrainingData(Base):
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)

class TestMapping(Base):
    __tablename__ = 'test_mapping'
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    delta_y = Column(Float)
    ideal_func_no = Column(Integer)

class DatabaseHandler:
    def __init__(self, db_path="data/ideal_functions.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def insert_training_data(self, dataframe):
        session = self.Session()
        try:
            for _, row in dataframe.iterrows():
                record = TrainingData(x=row[0], y1=row[1], y2=row[2], y3=row[3], y4=row[4])
                session.add(record)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def insert_test_mapping(self, dataframe):
        session = self.Session()
        try:
            for _, row in dataframe.iterrows():
                record = TestMapping(x=row["x"], y=row["y"], delta_y=row["delta_y"], ideal_func_no=row["ideal_func_no"])
                session.add(record)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
