@echo off

ctime -begin qwen.ctm

mkdir build >nul 2>&1
copy ".\external\lib\*.dll"  ".\build\" >nul 2>&1

:: set enviroment vars and requred stuff for the msvc compiler
call "D:\Programming\Software\msvc\setup_x64.bat"

set EXE_NAME=qwen
set TEST_NAME=qwen_test

:: /Bt /d2cgsummary 
set CFLAGS=/fp:fast /W4 /MD /nologo /utf-8 /std:c++latest /arch:AVX 
:: /SUBSYSTEM:CONSOLE /SUB-SYSTEM:WINDOWS /INCREMENTAL /time /Fe:%EXE_NAME%.exe
set L_FLAGS=/SUBSYSTEM:CONSOLE

set APP_SRC=..\src\main.cpp
set TEST_SRC=..\test\test.cpp 

set SRC=..\src\tokenizer.cpp

set INCLUDE_DIRS=/I..\inc /I..\external\inc\
set LIBRARY_DIRS=/LIBPATH:..\external\lib\
set LIBRARIES=Catch2.lib Catch2Main.lib pcre2-8-static.lib user32.lib shell32.lib kernel32.lib

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
    cl %CFLAGS% /O2 %INCLUDE_DIRS% %TEST_SRC% %SRC% /Fe:%TEST_NAME%.exe /link %LIBRARY_DIRS% %LIBRARIES% %L_FLAGS%
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

echo Unknown command: %1
exit /b 1

:build_release
cl %CFLAGS% /O2 %INCLUDE_DIRS% %APP_SRC% %SRC% /Fe:%~1.exe /link %LIBRARY_DIRS% %LIBRARIES% %L_FLAGS%
if errorlevel 1 (
    echo -----------------------------------------------------------------
    echo Build failed!
    echo -----------------------------------------------------------------
    exit /b 1
)
exit /b 0

:build_debug
cl /Zi %CFLAGS% /fsanitize=address %INCLUDE_DIRS% %APP_SRC% %SRC% /Fe:%~1.exe /link %LIBRARY_DIRS% %LIBRARIES% %L_FLAGS% /DEBUG
if errorlevel 1 (
    echo -----------------------------------------------------------------
    echo Build failed !
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
