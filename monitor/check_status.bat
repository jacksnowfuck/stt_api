@echo off
setlocal enabledelayedexpansion
rem ====== ���ò��� ======
set "url=http://127.0.0.1:9933/api"
set "audio=D:\aigc\stt_api\monitor\monitor.wav"
set "restart_script=D:\aigc\stt_api\start_stt.bat"
set "log_file=D:\aigc\stt_api\monitor\monitor.log"
set "curl_result=curl_result.txt"
set "curl_tmp=curl_tmp.log"

rem ====== ���ýӿڣ��� curl ���� POST ���� ======
curl -o NUL -s -w "%%{http_code} %%{time_total}" ^
    -X POST "%url%" ^
    -F "audio=@%audio%" ^
    -H "Content-Type: multipart/form-data" ^
    2> "%curl_tmp%" > "%curl_result%"

rem ====== ��ȡ curl ������ ======
set "curl_output="
for /f "usebackq delims=" %%A in ("%curl_result%") do (
    set "curl_output=%%A"
)

rem ====== ����״̬�루curl �����ʽ�� 200 0.123��=====
for /f "tokens=1,2" %%a in ("!curl_output!") do (
    set "status=%%a"
    set "time_total=%%b"
)

echo [%date% %time%] ���󷵻�״̬��: !status!, ��ʱ: !time_total!s >> "%log_file%"

rem ====== �ж�״̬�� ======
if "!status!"=="200" (
    echo [%date% %time%] ������������������� >> "%log_file%"
) else (
    echo [%date% %time%] ״̬���쳣��ִ�������ű��� >> "%log_file%"
    taskkill /F /IM python.exe /T >nul 2>&1
    call "%restart_script%"
)

rem ======�������======
del curl_result.txt
del curl_tmp.log

endlocal