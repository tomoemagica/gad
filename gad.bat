@echo off

for %%f in (data_src\*.png) do (
    basename %%f | py gad.py
)

pause