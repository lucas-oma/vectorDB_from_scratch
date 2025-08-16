#!/usr/bin/env python3
"""
Test runner for MongoDB tests
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def check_mongodb_connection():
    """Check if MongoDB is accessible."""
    try:
        import motor.motor_asyncio
        # Use the same MongoDB URI as the main application
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://admin:password@localhost:27017/vector_db?authSource=admin")
        client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_uri)
        # Try to connect
        client.admin.command('ping')
        print("✅ MongoDB connection successful")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False



def generate_test_data():
    """Generate test data using the data generator."""
    print("\n📊 Generating test data...")
    
    try:
        # Try to generate data with real embeddings
        cmd = [sys.executable, "tests-mongo/data_generator.py"]
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("✅ Test data generated successfully")
            return True
        else:
            print("⚠️  Failed to generate data with API, will use simple data")
            return False
    except Exception as e:
        print(f"⚠️  Error generating test data: {e}")
        return False

def run_e2e_tests():
    """Run end-to-end tests."""
    print("\n🚀 Running E2E tests...")
    
    # Set test environment variables
    env = os.environ.copy()
    # Use the same MongoDB configuration as the main application
    env["MONGODB_URI"] = os.getenv("MONGODB_URI", "mongodb://admin:password@localhost:27017/vector_db?authSource=admin")
    env["MONGODB_DB"] = os.getenv("MONGODB_DB", "vector_db")
    env["BASE_URL"] = os.getenv("BASE_URL", "http://localhost:8000/v1")
    
    # Run E2E tests
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "tests-mongo/test_e2e_mongo.py"
    ]
    
    try:
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent.parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ E2E test execution failed: {e}")
        return False

def main():
    """Main test runner."""
    print("🧪 MongoDB Test Suite")
    print("=" * 50)
    
    # Check MongoDB connection
    if not check_mongodb_connection():
        print("\n❌ Cannot proceed without MongoDB connection")
        print("Please ensure MongoDB is running and accessible")
        sys.exit(1)
    
    # Generate test data
    generate_test_data()
    
    # Run E2E tests
    e2e_tests_passed = run_e2e_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    if  e2e_tests_passed:
        print("✅ E2E tests passed")
    else:
        print("❌ E2E tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 