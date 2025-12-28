"""
Database CRUD operations with a Supabase-like interface.
This module provides a compatibility layer that mimics the Supabase client API.
"""
from typing import List, Dict, Any
from sqlalchemy.orm import Session
import uuid
from app.database.connection import SessionLocal
from app.database.models import User, OTP, Document, DocumentEmbedding, ChatHistory, Meeting, UserGoogleAuth


class QueryResult:
    """Mimics Supabase query result"""
    def __init__(self, data: List[Dict]):
        self.data = data


class QueryBuilder:
    """Mimics Supabase query builder interface"""

    def __init__(self, model_class):
        self.model_class = model_class
        self._filters = []
        self._select_columns = "*"
        self._order_by = None
        self._limit_val = None
        self._in_filters = []

    def _get_session(self) -> Session:
        """Get a fresh session for each operation"""
        return SessionLocal()

    def select(self, columns: str = "*") -> "QueryBuilder":
        self._select_columns = columns
        return self

    def eq(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append((column, "=", value))
        return self

    def neq(self, column: str, value: Any) -> "QueryBuilder":
        self._filters.append((column, "!=", value))
        return self

    def in_(self, column: str, values: List[Any]) -> "QueryBuilder":
        self._in_filters.append((column, values))
        return self

    def order(self, column: str, desc: bool = False) -> "QueryBuilder":
        self._order_by = (column, desc)
        return self

    def limit(self, count: int) -> "QueryBuilder":
        self._limit_val = count
        return self

    def _convert_uuid(self, value: Any) -> Any:
        """Convert string to UUID if applicable"""
        if isinstance(value, str):
            try:
                return uuid.UUID(value)
            except ValueError:
                pass
        return value

    def _apply_filters(self, query, session: Session):
        for column, op, value in self._filters:
            col_attr = getattr(self.model_class, column, None)
            if col_attr is not None:
                if column in ('id', 'user_id', 'document_id'):
                    value = self._convert_uuid(value)

                if op == "=":
                    query = query.filter(col_attr == value)
                elif op == "!=":
                    query = query.filter(col_attr != value)

        for column, values in self._in_filters:
            col_attr = getattr(self.model_class, column, None)
            if col_attr is not None:
                if column in ('id', 'user_id', 'document_id'):
                    values = [self._convert_uuid(v) for v in values]
                query = query.filter(col_attr.in_(values))

        return query

    def execute(self) -> QueryResult:
        session = self._get_session()
        try:
            query = session.query(self.model_class)
            query = self._apply_filters(query, session)

            if self._order_by:
                col_attr = getattr(self.model_class, self._order_by[0], None)
                if col_attr is not None:
                    if self._order_by[1]:
                        query = query.order_by(col_attr.desc())
                    else:
                        query = query.order_by(col_attr.asc())

            if self._limit_val:
                query = query.limit(self._limit_val)

            results = query.all()
            data = [r.to_dict() for r in results]
            return QueryResult(data)
        finally:
            session.close()

    def update(self, data: Dict[str, Any]) -> QueryResult:
        session = self._get_session()
        try:
            query = session.query(self.model_class)
            query = self._apply_filters(query, session)

            results = query.all()
            updated_data = []

            for obj in results:
                for key, value in data.items():
                    if hasattr(obj, key):
                        setattr(obj, key, value)
                session.add(obj)
                updated_data.append(obj.to_dict())

            session.commit()
            return QueryResult(updated_data)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete(self) -> QueryResult:
        session = self._get_session()
        try:
            query = session.query(self.model_class)
            query = self._apply_filters(query, session)

            results = query.all()
            deleted_data = [r.to_dict() for r in results]

            for obj in results:
                session.delete(obj)

            session.commit()
            return QueryResult(deleted_data)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


class InsertQueryBuilder(QueryBuilder):
    """Query builder for insert operations"""

    def __init__(self, model_class, insert_data: Dict[str, Any]):
        super().__init__(model_class)
        self._insert_data = insert_data

    def execute(self) -> QueryResult:
        session = self._get_session()
        try:
            data = self._insert_data.copy()

            # Handle UUID fields
            for key in ['id', 'user_id', 'document_id']:
                if key in data:
                    data[key] = self._convert_uuid(data[key])

            # Handle document_ids array - convert list of strings to list of UUIDs
            if 'document_ids' in data and data['document_ids']:
                data['document_ids'] = [self._convert_uuid(doc_id) for doc_id in data['document_ids']]

            # Handle reserved field name mappings
            if self.model_class.__tablename__ == 'document_embeddings' and 'metadata' in data:
                data['chunk_metadata'] = data.pop('metadata')
            if self.model_class.__tablename__ == 'chat_history' and 'model_config' in data:
                data['model_config_str'] = data.pop('model_config')

            obj = self.model_class(**data)
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return QueryResult([obj.to_dict()])
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


class UpdateQueryBuilder(QueryBuilder):
    """Query builder for update operations"""

    def __init__(self, model_class, update_data: Dict[str, Any]):
        super().__init__(model_class)
        self._update_data = update_data

    def execute(self) -> QueryResult:
        return self.update(self._update_data)


class DeleteQueryBuilder(QueryBuilder):
    """Query builder for delete operations"""

    def execute(self) -> QueryResult:
        return self.delete()


class TableProxy:
    """Proxy class for table operations"""

    def __init__(self, model_class):
        self.model_class = model_class

    def select(self, columns: str = "*") -> QueryBuilder:
        return QueryBuilder(self.model_class).select(columns)

    def insert(self, data: Dict[str, Any]) -> InsertQueryBuilder:
        return InsertQueryBuilder(self.model_class, data)

    def update(self, data: Dict[str, Any]) -> UpdateQueryBuilder:
        return UpdateQueryBuilder(self.model_class, data)

    def delete(self) -> DeleteQueryBuilder:
        return DeleteQueryBuilder(self.model_class)


class DatabaseClient:
    """Main database client that mimics Supabase client interface"""

    _tables = {
        "users": User,
        "otps": OTP,
        "documents": Document,
        "document_embeddings": DocumentEmbedding,
        "chat_history": ChatHistory,
        "meetings": Meeting,
        "user_google_auth": UserGoogleAuth
    }

    def table(self, name: str) -> TableProxy:
        model_class = self._tables.get(name)
        if model_class is None:
            raise ValueError(f"Unknown table: {name}")
        return TableProxy(model_class)


# Global database client instance
db_client = DatabaseClient()
