from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import threading
from typing import Generator
import json


class SessionManager:
    """Database session management"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, database_url: str = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SessionManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, database_url: str = None):
        if not self._initialized:
            if database_url is None:
                raise ValueError("Database URL must be provided")

            self.database_url = database_url
            self.engine = None
            self.session_factory = None
            self.scoped_session = None

            self._initialize_engine()
            self._initialized = True

    def _initialize_engine(self):
        """Initialize database engine"""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=3600,
                echo=False,  # Set to True for SQL logging
                future=True
            )

            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )

            # Create scoped session for thread safety
            self.scoped_session = scoped_session(self.session_factory)

            print(f"✅ Database engine initialized: {self.database_url}")

        except Exception as e:
            print(f"❌ Error initializing database engine: {e}")
            raise

    @contextmanager
    def session_scope(self) -> Generator:
        """Provide a transactional scope around a series of operations"""
        session = self.scoped_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Session error: {e}")
            raise
        finally:
            session.close()

    def get_session(self):
        """Get a new database session"""
        return self.scoped_session()

    def close_session(self, session):
        """Close a database session"""
        if session:
            session.close()

    def close_all_sessions(self):
        """Close all database sessions"""
        if self.scoped_session:
            self.scoped_session.remove()

    def create_tables(self, base):
        """Create all tables"""
        try:
            base.metadata.create_all(self.engine)
            print("✅ Database tables created")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            raise

    def drop_tables(self, base):
        """Drop all tables"""
        try:
            base.metadata.drop_all(self.engine)
            print("✅ Database tables dropped")
        except Exception as e:
            print(f"❌ Error dropping tables: {e}")
            raise

    def check_connection(self) -> bool:
        """Check database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False

    def get_connection_info(self) -> dict:
        """Get database connection information"""
        info = {
            'url': str(self.engine.url),
            'pool_size': self.engine.pool.size(),
            'pool_overflow': self.engine.pool.overflow(),
            'pool_checked_in': self.engine.pool.checkedin(),
            'pool_checked_out': self.engine.pool.checkedout()
        }
        return info

    def optimize_connection_pool(self):
        """Optimize connection pool settings"""
        try:
            # Disconnect all connections
            self.engine.dispose()

            # Recreate engine with optimized settings
            self._initialize_engine()

            print("✅ Connection pool optimized")
        except Exception as e:
            print(f"❌ Error optimizing connection pool: {e}")

    def execute_raw_sql(self, sql: str, params: dict = None):
        """Execute raw SQL statement"""
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(sql, params)
                else:
                    result = conn.execute(sql)

                if result.returns_rows:
                    return [dict(row) for row in result]
                else:
                    return {"rowcount": result.rowcount}
        except Exception as e:
            print(f"❌ Error executing SQL: {e}")
            raise

    def bulk_insert(self, table_name: str, data: list):
        """Bulk insert data"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    f"INSERT INTO {table_name} VALUES (:values)",
                    [{"values": json.dumps(row)} for row in data]
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"❌ Error bulk inserting: {e}")
            return False

    def transaction(self):
        """Create a transaction context"""
        return self.engine.begin()

    def vacuum_database(self):
        """Vacuum database (SQLite specific)"""
        if 'sqlite' in self.database_url:
            try:
                with self.engine.connect() as conn:
                    conn.execute("VACUUM")
                print("✅ Database vacuumed")
            except Exception as e:
                print(f"❌ Error vacuuming database: {e}")

    def backup_database(self, backup_path: str):
        """Backup database"""
        try:
            if 'sqlite' in self.database_url:
                import shutil
                import os

                # Close all connections
                self.close_all_sessions()

                # Copy database file
                db_path = self.database_url.replace('sqlite:///', '')
                shutil.copy2(db_path, backup_path)

                print(f"✅ Database backed up to {backup_path}")
            else:
                print("⚠️ Backup only supported for SQLite databases")
        except Exception as e:
            print(f"❌ Error backing up database: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close_all_sessions()
            if self.engine:
                self.engine.dispose()
        except:
            pass