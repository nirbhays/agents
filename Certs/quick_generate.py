#!/usr/bin/env python3
"""
Quick PDF Generator - Single command to regenerate all certification PDFs
Usage: python quick_generate.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 Quick PDF Generator")
    print("=" * 50)
    
    script_path = Path(__file__).parent / "generate_professional_pdfs.py"
    
    if not script_path.exists():
        print("❌ Error: generate_cert_pdfs.py not found!")
        sys.exit(1)
    
    print("📄 Generating PDFs...")
    print()
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              check=True, 
                              capture_output=False,
                              text=True)
        print()
        print("✅ Done! Check the pdf_output folder.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating PDFs: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
