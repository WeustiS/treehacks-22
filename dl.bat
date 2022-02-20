@echo off
set /p netID="Input NetID:"

:: EDIT THESE TO CHANGE STUFF
:: Unit seconds
set /a timeWait=0
:: Unit seconds OR if > 60s then format is HH:MM:SS.000
set /a x=0
set /a dummy=%x% %% (21600/%timeWait%)
echo Hello, %netID%! We are going to get screenshots. We will wait %timeWait% seconds between screenshots.  
set /p t="Okay? (Enter or Control+C)"
:grab_url
if %dummy% == 0 youtube-dl -f 96 -g h6hzVOwaN_4 > url.log else (echo NoNewLog)
set /P Url=<url.log
call :GetUnixTime UNIX_TIME
echo %Url%
ffmpeg -i %Url% -vframes 1 -q:v 5  %netID%_%UNIX_TIME%_%x%.jpg
set /a x+=1
echo Waiting%timeWait%Seconds
ping -4 -n %timeWait% "">nul
echo StartingVideo%x%!
goto grab_url

:GetUnixTime
setlocal enableextensions
for /f %%x in ('wmic path win32_utctime get /format:list ^| findstr "="') do (
    set %%x)
set /a z=(14-100%Month%%%100)/12, y=10000%Year%%%10000-z
set /a ut=y*365+y/4-y/100+y/400+(153*(100%Month%%%100+12*z-3)+2)/5+Day-719469
set /a ut=ut*86400+100%Hour%%%100*3600+100%Minute%%%100*60+100%Second%%%100
endlocal & set "%1=%ut%" & goto :EOF
