# """
# main.py
# -------
# This file serves as the single entry point for running
# the Password Strength Analyzer web application.

# Instead of running app.py directly, reviewers or users
# can run this file to start the Flask server.

# Project : Password Strength Analyzer using Machine Learning
# """

# import os
# import sys

# # -------------------------------------------------
# # Ensure project root is in Python path
# # -------------------------------------------------

# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# # -------------------------------------------------
# # Import Flask app
# # -------------------------------------------------

# try:
#     from app import app
# except ImportError as e:
#     raise ImportError(
#         f"Failed to import Flask app. Check project structure. Error: {e}"
#     )

# # -------------------------------------------------
# # Run Application
# # -------------------------------------------------

# def main():
#     """
#     Main function to run the Flask web application
#     """
#     print("🚀 Starting Password Strength Analyzer Web Application...")
#     print("🌐 Access URL: http://127.0.0.1:5000")

#     app.run(
#         host="0.0.0.0",
#         port=5000,
#         debug=True
#     )


# # -------------------------------------------------
# # Program Entry Point
# # -------------------------------------------------

# if __name__ == "__main__":
#     main()














"""
main.py
-------
Entry point to run the Password Strength Analyzer web application.

Project : Password Strength Analyzer using Machine Learning
"""

import os
import sys

# -------------------------------------------------
# Ensure project root is in Python path
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# -------------------------------------------------
# Import Flask app
# -------------------------------------------------
try:
    from app import app  # app.py should define 'app = Flask(__name__)'
except ImportError as e:
    raise ImportError(
        f"Failed to import Flask app. Check project structure. Error: {e}"
    )

# -------------------------------------------------
# Run Application
# -------------------------------------------------
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

# -------------------------------------------------
# Program Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    main()


