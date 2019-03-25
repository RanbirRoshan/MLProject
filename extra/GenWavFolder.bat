@echo off
set input_folder=%~1
set output_folder=%1%_wav
rmdir %output_folder%
mkdir %output_folder%
call :WalkFolderAndConvert %input_folder%
EXIT /B %ERRORLEVEL%



:WalkFolderAndConvert
for /r %%f in (*) do call "C:\Program Files (x86)\sox-14-4-2\sox" %%f %%f.wav
EXIT /B 0