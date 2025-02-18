#!/bin/bash
echo "Stopping Docker containers..."
docker-compose down -v

echo "Cleaning up database files..."
rm -f anarchy_copilot.db
docker run --rm -v $(pwd):/app -w /app alpine rm -f /app/anarchy_copilot.db

echo "Starting Docker containers..."
docker-compose up -d

echo "Waiting for dev container to be ready..."
until docker exec anarchy-copilot_dev_1 echo "Container is running" > /dev/null 2>&1; do
    echo "Waiting for dev container..."
    sleep 2
done

echo "Initializing database..."
docker exec anarchy-copilot_dev_1 alembic upgrade head

echo "Creating test user..."
docker exec anarchy-copilot_dev_1 python3 scripts/create_test_user.py

echo "Database reset complete!"
