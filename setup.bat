@echo off
rem ============================================================================================================
rem  setup.bat - one-step environment setup for the floor-plan detection project (Windows).
rem
rem    setup.bat            CPU installation (works on any machine, no GPU needed)
rem    setup.bat gpu        additionally installs the NVIDIA GPU packages (only if nvidia-smi finds a GPU; ~3 GB download)
rem
rem  Creates floor-plan-detection\.venv (Python 3.11), installs the pinned packages, and runs a quick self-check.
rem  Needs an internet connection once (packages). Afterwards everything runs offline (OCR models ship inside the package).
rem ============================================================================================================
setlocal
set "ROOT=%~dp0"
set "PROJ=%ROOT%floor-plan-detection"
set "VENV=%PROJ%\.venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PROJ%\requirements.txt" (
  echo [setup] Cannot find "%PROJ%\requirements.txt". Run setup.bat from the folder that contains floor-plan-detection.
  exit /b 1
)

rem ---- 1. uv (fast installer that can also fetch Python 3.11 by itself) ----------------------------------------
where uv >nul 2>&1
if errorlevel 1 (
  echo [setup] 'uv' not found - trying to install it with winget...
  winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements >nul 2>&1
  where uv >nul 2>&1
  if errorlevel 1 (
    echo [setup] winget did not provide uv - trying pip...
    python -m pip install --user uv >nul 2>&1
    where uv >nul 2>&1
    if errorlevel 1 (
      python -m uv --version >nul 2>&1
      if errorlevel 1 (
        echo [setup] Could not install 'uv'. Install it manually from https://docs.astral.sh/uv/ and run setup.bat again.
        exit /b 1
      )
      set "UV=python -m uv"
    ) else (
      set "UV=uv"
    )
  ) else (
    set "UV=uv"
  )
) else (
  set "UV=uv"
)
echo [setup] using: %UV%

rem ---- 2. virtual environment -----------------------------------------------------------------------------------
if not exist "%PY%" (
  echo [setup] creating Python 3.11 environment in "%VENV%" ...
  %UV% venv --python 3.11 "%VENV%"
  if errorlevel 1 (echo [setup] venv creation failed & exit /b 1)
) else (
  echo [setup] environment already exists - reusing it
)

rem ---- 3. packages ----------------------------------------------------------------------------------------------
echo [setup] installing pinned packages (CPU) ...
%UV% pip install --python "%PY%" -r "%PROJ%\requirements.txt"
if errorlevel 1 (echo [setup] package installation failed & exit /b 1)
%UV% pip install --python "%PY%" --no-deps rapidocr-onnxruntime==1.4.4
if errorlevel 1 (echo [setup] rapidocr installation failed & exit /b 1)

rem ---- 4. optional GPU ------------------------------------------------------------------------------------------
if /i "%~1"=="gpu" (
  nvidia-smi >nul 2>&1
  if errorlevel 1 (
    echo [setup] 'gpu' requested but nvidia-smi found no NVIDIA GPU - staying on the CPU installation.
  ) else (
    echo [setup] NVIDIA GPU found - installing PyTorch cu128 and onnxruntime-gpu 1.23.2 ^(large download^) ...
    %UV% pip uninstall --python "%PY%" onnxruntime
    %UV% pip install --python "%PY%" -r "%PROJ%\requirements-gpu.txt"
    if errorlevel 1 (echo [setup] GPU packages failed - the CPU installation still works with --device cpu)
  )
)

rem ---- 5. self-check --------------------------------------------------------------------------------------------
echo [setup] self-check ...
pushd "%PROJ%"
"%PY%" -c "import cv2, numpy, shapely, pymupdf, skimage, yaml, rapidocr_onnxruntime, onnxruntime; from src.gpu import resolve_device; print('imports OK | device that --device auto will use:', resolve_device('auto')[0])"
if errorlevel 1 (popd & echo [setup] self-check failed & exit /b 1)
"%PY%" -m pytest tests/test_shapes.py tests/test_raster_segmentation.py -q
if errorlevel 1 (popd & echo [setup] unit tests failed & exit /b 1)
popd

echo.
echo [setup] DONE. Run a floor plan with:
echo     run.bat "your_plan.pdf" "output_folder"
echo See HANDOFF.md for the full project state.
endlocal
exit /b 0
