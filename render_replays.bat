@echo off
echo Starting Replay Generation...
echo This will launch TrackMania and simulate the runs.
echo Please do not touch the mouse or keyboard!
"C:\Users\felix\Documents\linesight\venv\Scripts\python.exe" scripts/tools/video_stuff/inputs_to_gbx.py --inputs_dir "%~dp0replays_to_render" --map_path "ESL-Hockolicious.Challenge.Gbx" --cutoff_time 70
echo Done! Check replays_to_render for .Replay.Gbx files.
pause
