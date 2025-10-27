@echo off
setlocal enabledelayedexpansion
rem ====== 配置部分 ======
set "url=http://127.0.0.1:9933/api"
set "audio=D:\aigc\stt_api\monitor\monitor.wav"
set "restart_script=D:\aigc\stt_api\start_stt.bat"
set "log_file=D:\aigc\stt_api\monitor\monitor.log"
set "curl_result=curl_result.txt"
set "curl_tmp=curl_tmp.log"

rem ====== 调用接口：用 curl 发送 POST 请求 ======
curl -o NUL -s -w "%%{http_code} %%{time_total}" ^
    -X POST "%url%" ^
    -F "audio=@%audio%" ^
    -H "Content-Type: multipart/form-data" ^
    2> "%curl_tmp%" > "%curl_result%"

rem ====== 读取 curl 输出结果 ======
set "curl_output="
for /f "usebackq delims=" %%A in ("%curl_result%") do (
    set "curl_output=%%A"
)

rem ====== 解析状态码（curl 输出格式： 200 0.123）=====
for /f "tokens=1,2" %%a in ("!curl_output!") do (
    set "status=%%a"
    set "time_total=%%b"
)

echo [%date% %time%] 请求返回状态码: !status!, 耗时: !time_total!s >> "%log_file%"

rem ====== 判断状态码 ======
if "!status!"=="200" (
    echo [%date% %time%] 服务正常，无需操作。 >> "%log_file%"
) else (
    echo [%date% %time%] 状态码异常，执行重启脚本。 >> "%log_file%"
    taskkill /F /IM python.exe /T >nul 2>&1
    call "%restart_script%"
)

rem ======清楚缓存======
del curl_result.txt
del curl_tmp.log

endlocal