@echo off

for %%f in (data_src\*.png) do (
    echo %%f | py gad.py
)

pause