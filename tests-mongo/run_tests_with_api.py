#!/usr/bin/env python3
"""
Test runner that starts a dedicated test API service, runs tests, and cleans up
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Load environment variables from .env file in the root directory
try:
    from dotenv import load_dotenv
    load_dotenv("../.env")
except ImportError:
    pass

def start_test_api():
    """Start the test API service."""
    print("🚀 Starting test API service...")
    try:
        # Start the test API service
        cmd = ["docker-compose", "-f", "docker-compose.test.yml", "up", "-d"]
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        
        if result.returncode != 0:
            print("❌ Failed to start test API service")
            return False
        
        # Wait for services to be healthy
        print("⏳ Waiting for services to be ready...")
        time.sleep(30)  # Give time for health checks
        
        # Check if API is responding
        import httpx
        try:
            response = httpx.get("http://localhost:8001/v1/healthz", timeout=10)
            if response.status_code == 200:
                print("✅ Test API service is ready")
                return True
            else:
                print(f"❌ Test API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Test API not responding: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting test API: {e}")
        return False

def stop_test_api():
    """Stop the test API service."""
    print("🛑 Stopping test API service...")
    try:
        cmd = ["docker-compose", "-f", "docker-compose.test.yml", "down"]
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            print("✅ Test API service stopped")
        else:
            print("⚠️  Warning: Failed to stop test API service")
            
    except Exception as e:
        print(f"⚠️  Warning: Error stopping test API: {e}")

def run_tests():
    """Run the actual tests."""
    print("\n🧪 Running tests...")
    
    # Set test environment variables
    env = os.environ.copy()
    env["TEST_BASE_URL"] = "http://localhost:8001/v1"
    
    # Run all test files
    test_files = [
        "test_e2e_mongo.py",
        "test_ivf_e2e.py", 
        "test_lsh_simhash_e2e.py",
        "test_persistence.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}...")
        cmd = [
            sys.executable, "-m", "pytest",
            "-v",
            "--tb=short",
            "--asyncio-mode=auto",
            test_file
        ]
        
        try:
            result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent)
            if result.returncode != 0:
                all_passed = False
                print(f"❌ {test_file} failed")
            else:
                print(f"✅ {test_file} passed")
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Main test runner."""
    print("🧪 VectorDB Test Suite with Dedicated Test API")
    print("=" * 60)
    
    # Start test API
    if not start_test_api():
        print("\n❌ Cannot proceed without test API")
        sys.exit(1)
    
    try:
        # Run tests
        tests_passed = run_tests()
        
        # Print summary
        print("\n" + "=" * 60)
        if tests_passed:
            print("🎉 All tests passed!")
        else:
            print("❌ Some tests failed")
            sys.exit(1)
            
    finally:
        # Always stop the test API
        stop_test_api()

if __name__ == "__main__":
    main() 