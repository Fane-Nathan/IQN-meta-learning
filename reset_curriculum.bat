@echo off
echo ========================================
echo   Linesight Curriculum Reset Utility
echo ========================================
echo.

REM Delete old curriculum config
if exist curriculum_config.txt (
    echo Deleting stale curriculum_config.txt...
    del curriculum_config.txt
)

REM Delete saved states directory
if exist states (
    echo Deleting saved states directory...
    rmdir /s /q states
)

REM Create fresh config with safe defaults
echo Creating fresh curriculum_config.txt...
echo FORCE_SPAWN_ZONE=None> curriculum_config.txt
echo FOCUS_ZONE=None>> curriculum_config.txt

echo.
echo Done! Curriculum reset to clean state.
echo You can now run launch_training.bat
pause