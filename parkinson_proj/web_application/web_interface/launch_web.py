#!/usr/bin/env python3
"""
Launcher script for the Interactive Video Analysis Web GUI.
"""

import os
import sys
import argparse

# Add the parkinson_proj directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Interactive Video Analysis Web App Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the web app")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug mode")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to custom model")
    
    return parser.parse_args()

def main():
    """Launch the web application"""
    args = parse_args()
    
    try:
        from app import main as app_main
        
        print("🚀 Launching Interactive Video Analysis Web App...")
        print("=" * 50)
        print(f"📱 Web interface will be available at: http://{args.host}:{args.port}")
        print(f"🐛 Debug mode: {'Enabled' if args.debug else 'Disabled'}")
        if args.model_path:
            print(f"🤖 Custom model: {args.model_path}")
        print("🔄 Loading...")
        
        # Pass arguments to the app main function
        app_main(port=args.port, host=args.host, debug=args.debug, model_path=args.model_path)
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you're running this from the correct directory")
        print("📁 Current directory:", os.getcwd())
        
    except Exception as e:
        print(f"❌ Failed to launch web app: {e}")
        print("💡 Try running: python app.py")

if __name__ == "__main__":
    main() 