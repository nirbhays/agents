@echo off
REM Quick PDF Generator - Windows Batch Script
REM Double-click this file to regenerate all PDFs

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║     GCP Certification PDF Generator                      ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

python generate_professional_pdfs.py

echo.
echo ═══════════════════════════════════════════════════════════
echo.
echo Press any key to open the PDF folder...
pause > nul

explorer pdf_output

echo.
echo Done! Press any key to exit.
pause > nul
