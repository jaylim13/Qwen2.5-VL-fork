#!/usr/bin/env python3
"""
Main web application entry point for Qwen2.5-VL Parkinson's Project.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL Parkinson's Project Web Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--interface", type=str, choices=["web", "cli"], default="web",
                       help="Interface type to launch")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port for web interface")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host for web interface")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Enable debug mode")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to custom model")
    
    return parser.parse_args()

def launch_web_interface(port, host, debug, model_path):
    """Launch the web interface"""
    try:
        # Import the web app
        from parkinson_proj.web_application.web_interface.app import main as web_main
        
        print("🚀 Launching Web Interface...")
        print("=" * 50)
        print(f"📱 URL: http://{host}:{port}")
        print(f"🐛 Debug: {'Enabled' if debug else 'Disabled'}")
        if model_path:
            print(f"🤖 Model: {model_path}")
        print("=" * 50)
        
        # Launch the web app
        web_main()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure the web application is properly installed")
        return False
    except Exception as e:
        print(f"❌ Failed to launch web interface: {e}")
        return False
    
    return True

def launch_cli_interface(debug, model_path):
    """Launch the CLI interface"""
    try:
        # Import the CLI app
        from parkinson_proj.web_application.cli.cli_app import main as cli_main
        
        print("🚀 Launching CLI Interface...")
        print("=" * 50)
        print(f"🐛 Debug: {'Enabled' if debug else 'Disabled'}")
        if model_path:
            print(f"🤖 Model: {model_path}")
        print("=" * 50)
        
        # Launch the CLI app
        cli_main()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure the CLI application is properly installed")
        return False
    except Exception as e:
        print(f"❌ Failed to launch CLI interface: {e}")
        return False
    
    return True

def main():
    """Main entry point"""
    args = parse_args()
    
    print("🧠 Qwen2.5-VL Parkinson's Project")
    print("=" * 50)
    print(f"🖥️  Interface: {args.interface.upper()}")
    print(f"📁 Project Root: {PROJECT_ROOT}")
    
    success = False
    
    if args.interface == "web":
        success = launch_web_interface(args.port, args.host, args.debug, args.model_path)
    elif args.interface == "cli":
        success = launch_cli_interface(args.debug, args.model_path)
    
    if not success:
        print("\n❌ Failed to launch the application")
        print("💡 Try checking the installation and dependencies")
        sys.exit(1)

if __name__ == "__main__":
    main()