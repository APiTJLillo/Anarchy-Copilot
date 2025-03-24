import asyncio
from database import AsyncSessionLocal
from sqlalchemy import text

async def check_db():
    async with AsyncSessionLocal() as session:
        # Check tables
        result = await session.execute(text('SELECT name FROM sqlite_master WHERE type="table"'))
        tables = [row[0] for row in result]
        print("Tables:", tables)
        
        # Check active sessions
        if 'proxy_sessions' in tables:
            result = await session.execute(text('SELECT id, name, is_active FROM proxy_sessions WHERE is_active = 1'))
            sessions = [{"id": row[0], "name": row[1], "is_active": row[2]} for row in result]
            print("\nActive sessions:", sessions)
        
        # Check proxy history if exists
        if 'proxy_history' in tables:
            result = await session.execute(text('SELECT COUNT(*) FROM proxy_history'))
            count = result.scalar()
            print("\nProxy history entries:", count)
            
            # Show table schema
            result = await session.execute(text("""
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name='proxy_history'
            """))
            schema = result.scalar()
            print("\nProxy history schema:")
            print(schema)
            
            # Show last 5 entries if any
            if count > 0:
                result = await session.execute(text("""
                    SELECT id, session_id, method, url, timestamp, request_headers, status_code
                    FROM proxy_history ORDER BY id DESC LIMIT 5
                """))
                entries = result.fetchall()
                print("\nLast 5 entries:")
                for entry in entries:
                    print(f"ID: {entry[0]}, Session: {entry[1]}, Method: {entry[2]}, URL: {entry[3]}")
                    print(f"Timestamp: {entry[4]}, Headers: {entry[5]}, Status: {entry[6]}")
                    print("-" * 80)

if __name__ == "__main__":
    asyncio.run(check_db())
