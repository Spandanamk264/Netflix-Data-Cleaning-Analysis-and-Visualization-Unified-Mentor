import uvicorn
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

if __name__ == "__main__":
    print("🚀 Starting Netflix ML API...")
    print("📄 Documentation available at: http://localhost:8000/docs")
    
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
