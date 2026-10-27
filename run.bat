@echo off
rem Usage: run.bat "path\to\floorplan.pdf" [output_folder] [auto|cpu|cuda] [profile]
rem Relative paths are resolved against the folder you run this from (NOT the project folder).
rem profile = a sheet profile: a name from floor-plan-detection\config\profiles (e.g. a1800_school, student_accommodation) or a path to a yaml file.
rem           Needed for vector sheets whose printed scale is not 1:100 or that rely on CAD-symbol exemplars; raster PDFs need none.
if "%~1"=="" (echo Usage: run.bat "path\to\floorplan.pdf" [output_folder] [auto^|cpu^|cuda] [profile] & exit /b 1)
if not exist "%~1" (echo Input not found: %~f1 & exit /b 1)
set "IN=%~f1"
set "OUT=%~f2"
set "DEV=%~3"
if "%DEV%"=="" set "DEV=auto"
set "PROJ=%~dp0floor-plan-detection"
set "PROF="
if "%~4"=="" goto noprofile
set "PROF=%PROJ%\config\profiles\%~4.yaml"
if exist "%~f4" set "PROF=%~f4"
if not exist "%PROF%" (echo Profile not found: %PROF% & exit /b 1)
:noprofile
set "PYTHONIOENCODING=utf-8"
set "ARGS=--device %DEV%"
if not "%OUT%"=="" set ARGS=%ARGS% --out "%OUT%"
if not "%PROF%"=="" set ARGS=%ARGS% --profile "%PROF%"
pushd "%PROJ%"
".venv\Scripts\python.exe" -m src.cli "%IN%" %ARGS%
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
