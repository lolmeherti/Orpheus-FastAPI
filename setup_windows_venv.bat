@echo off
echo Creating Python 3.10 virtual environment...
py -3.10 -m venv orpheus-venv

echo Activating environment...
call orpheus-venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA 12.8 support...
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo Installing project dependencies...
pip install -r requirements-windows.txt

echo Done. You can now run:
echo.
echo     orpheus-venv\Scripts\activate.bat
echo     python conversation_windows.py