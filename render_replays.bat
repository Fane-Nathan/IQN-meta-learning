@echo off
echo 🎥 Starting Replay Rendering...
echo This will launch Trackmania and render 50 replays.
echo Please do not touch the mouse or keyboard!
"C:\Users\felix\Documents\linesight\venv\Scripts\python.exe" scripts/tools/video_stuff/inputs_to_gbx.py --inputs_dir replays_to_render --map_path "ESL-Hockolicious.Challenge.Gbx" --cutoff_time 65
pause
