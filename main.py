

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from app import app  # app.py should define 'app = Flask(__name__)'
except ImportError as e:
    raise ImportError(
        f"Failed to import Flask app. Check project structure. Error: {e}"
    )

def main():
    """
    Run the Flask web application
    """
    print("🚀 Starting Password Strength Analyzer Web Application...")
    print("🌐 Access URL: http://127.0.0.1:5000")
    print("🔧 Debug mode is ON for development purposes")

    app.run(
        host="0.0.0.0",  # accessible from local network
        port=5000,
        debug=True  # set False in production
    )
    
if __name__ == "__main__":
    main()


