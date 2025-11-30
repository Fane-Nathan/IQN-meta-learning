@echo off
REM ============================================
REM Progressive Spawn Zone Advancement
REM ============================================
REM Use this to force the AI to train on later track sections
REM Run BETWEEN training sessions

set CONFIG_FILE=curriculum_config.txt

echo.
echo ========================================
echo   CURRICULUM SPAWN ZONE MANAGER
echo ========================================
echo.

if "%1"=="" (
    echo Current config:
    echo ----------------
    if exist %CONFIG_FILE% (
        type %CONFIG_FILE%
    ) else (
        echo No config file found - using defaults
    )
    echo.
    echo Usage:
    echo   %0 [zone]     - Set spawn zone
    echo   %0 reset      - Reset to Zone 0
    echo   %0 auto       - Auto-advance based on progress
    echo.
    echo Recommended progression:
    echo   Phase 1: Zone 0    ^(default^)
    echo   Phase 2: Zone 1000 ^(after mastering 0-1200^)
    echo   Phase 3: Zone 2500 ^(after mastering 1000-3000^)
    echo   Phase 4: Zone 4000 ^(after mastering 2500-4500^)
    echo   Phase 5: Zone 5500 ^(after mastering 4000-6000^)
    echo   Phase 6: Zone 7000 ^(final stretch^)
    echo.
    goto :EOF
)

if "%1"=="reset" (
    echo FORCE_SPAWN_ZONE=None > %CONFIG_FILE%
    echo FOCUS_ZONE=None >> %CONFIG_FILE%
    echo.
    echo ✓ Reset to default spawning ^(Zone 0^)
    goto :EOF
)

if "%1"=="auto" (
    echo Auto-advancement not implemented in batch.
    echo Use: python curriculum_manager_v2.py --run
    goto :EOF
)

REM Set specific zone
echo FORCE_SPAWN_ZONE=%1 > %CONFIG_FILE%
echo FOCUS_ZONE=None >> %CONFIG_FILE%

echo.
echo ✓ Spawn zone set to: %1
echo.
echo IMPORTANT: Restart training for changes to take effect!
echo.

:EOF
