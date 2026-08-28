@echo off

ctime -begin qwen.ctm

mkdir build >nul 2>&1
copy ".\external\lib\*.dll"  ".\build\" >nul 2>&1

call "D:\Programming\Software\msvc\setup_x64.bat"

set CXX=clang++
set TARGET=--target=x86_64-pc-windows-msvc

set EXE_NAME=qwen
set TEST_NAME=qwen_test

set CFLAGS=%TARGET% -ffast-math -Wall -Wextra -fms-runtime-lib=dll -finput-charset=UTF-8 -mavx -mfma -fuse-ld=lld
:: -Wl,/SUBSYSTEM:CONSOLE  /  -Wl,/SUBSYSTEM:WINDOWS  /  -Wl,/INCREMENTAL  /  -DTRACY_ENABLE
set L_FLAGS=-Xlinker /SUBSYSTEM:CONSOLE

set APP_SRC=..\src\main.cpp
set TEST_SRC=..\test\test.cpp

set SRC_CPP=..\src\tokenizer.cpp ..\src\qwen_tables.cpp ..\src\gguf.cpp ..\src\engine.cpp ..\src\utils.cpp ..\external\inc\imgui\imgui_demo.cpp
set SRC_C=..\src\arena.c ..\external\src\glad.c

set INCLUDE_DIRS=-I..\inc -I..\external\inc\
set LIBRARY_DIRS=-L..\external\lib\

set LIBRARIES=-limgui -lglfw3 -lnfd -llunasvg -lplutovg -lpcre2-8-static -lutf8proc_static -lgdi32 -lole32 -luser32 -lshell32 -lkernel32
set TEST_LIBS=-lsimdjson -lCatch2 -lCatch2Main
set PROFILE_LIBS=-lTracyClient

if "%1"=="" (
    echo Usage:Specify a flag run.bat dbg ^| rel ^| all
    exit /b 1
)

if "%1"=="rel" (
    echo Building the project...
    pushd .\build
    call :build_release %EXE_NAME%
    if errorlevel 1 goto :build_failed
    echo -----------------------------------------------------------------
    echo Release Build Successful.
    echo Build time :
    ctime -end ../qwen.ctm
    :: call :cloc
    echo Running...
    echo -----------------------------------------------------------------
    .\%EXE_NAME%.exe
    goto :build_success
)

if "%1"=="dbg" (
    echo Building the project with debugging symbols...
    pushd .\build
    call :build_debug %EXE_NAME%
    if errorlevel 1 goto :build_failed
    echo -----------------------------------------------------------------
    echo Debug Build Successful.
    echo Build time :
    ctime -end ../qwen.ctm
    echo Launching debugger...
    echo -----------------------------------------------------------------
    start "" "raddbg.exe" .\%EXE_NAME%.exe
    :: start "" "devenv.exe" .\build\%EXE_NAME%.exe
    goto :build_success
)

if "%1"=="all" (
    echo Building..
    pushd .\build
    echo Building release [%EXE_NAME%.exe]...
    call :build_release %EXE_NAME%
    if errorlevel 1 goto :build_failed
    echo -----------------------------------------------------------------
    echo Building debug [%EXE_NAME%_dbg.exe]...
    call :build_debug %EXE_NAME%_dbg
    if errorlevel 1 goto :build_failed
    echo -----------------------------------------------------------------
    echo Build Successful.
    echo Build time :
    ctime -end ../qwen.ctm
    :: call :cloc
    echo Running release build...
    echo -----------------------------------------------------------------
    .\%EXE_NAME%.exe
    goto :build_success
)

if "%1"=="test" (
    echo Building Tests...
    pushd .\build
    %CXX% %CFLAGS% -O2 %INCLUDE_DIRS% -x c %SRC_C% -x c++ -std=c++26 %TEST_SRC% %SRC_CPP% -o %TEST_NAME%.exe %LIBRARY_DIRS% %LIBRARIES% %TEST_LIBS% %L_FLAGS%
    if errorlevel 1 goto :build_failed
    echo -----------------------------------------------------------------
    echo Tests Build Successful.
    echo Build time :
    ctime -end ../qwen.ctm
    :: call :cloc
    echo Running...
    echo -----------------------------------------------------------------
    .\%TEST_NAME%.exe
    goto :build_success
)

if "%1"=="prof" (
    echo Building release with Tracy profiling...
    pushd .\build
    call :build_profile %EXE_NAME%
    if errorlevel 1 goto :build_failed

    echo -----------------------------------------------------------------
    echo Tracy Profile Build Successful.
    echo Build time :
    ctime -end ../qwen.ctm

    echo -----------------------------------------------------------------
    echo Starting Tracy...
    start "" "D:\Programming\Software\Tracy\tracy-profiler.exe" -a 127.0.0.1

    echo -----------------------------------------------------------------
    echo Running profiled application...
    echo -----------------------------------------------------------------
    .\%EXE_NAME%.exe

    goto :build_success
)

echo Unknown command: %1
exit /b 1

:build_release
%CXX% %CFLAGS% -O2 -mavx2 -ffast-math %INCLUDE_DIRS% -std=c++26 %SRC_C% %APP_SRC% %SRC_CPP% -o %~1.exe %LIBRARY_DIRS% %LIBRARIES% %L_FLAGS%
if errorlevel 1 (
    echo -----------------------------------------------------------------
    echo Build failed!
    echo -----------------------------------------------------------------
    exit /b 1
)
exit /b 0

:build_debug
%CXX% -g -gcodeview %CFLAGS% -fsanitize=address %INCLUDE_DIRS% -std=c++26 %SRC_C% %APP_SRC% %SRC_CPP% -o %~1.exe %LIBRARY_DIRS% %LIBRARIES% %L_FLAGS% -Xlinker /DEBUG
if errorlevel 1 (
    echo -----------------------------------------------------------------
    echo Build failed !
    echo -----------------------------------------------------------------
    exit /b 1
)
exit /b 0

:build_profile
%CXX% %CFLAGS% -O2 -mavx2 -ffast-math -DTRACY_ENABLE %INCLUDE_DIRS% -std=c++26 %SRC_C% %APP_SRC% %SRC_CPP% -o %~1.exe %LIBRARY_DIRS% %LIBRARIES% %PROFILE_LIBS% %L_FLAGS%
if errorlevel 1 (
    echo -----------------------------------------------------------------
    echo Profile build failed!
    echo -----------------------------------------------------------------
    exit /b 1
)
exit /b 0

:cloc
D:\Programming\Software\cloc\cloc.exe --quiet --hide-rate ..\src\ ..\Main.c | more +2
exit /b 0

:build_failed
popd
exit /b 1

:build_success
popd
exit /b 0