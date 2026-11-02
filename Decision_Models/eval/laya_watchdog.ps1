$log="eval\outputs\laya_watchdog_log.md"
while($true){
 $p=@(Get-CimInstance Win32_Process -Filter "name like 'python%'" | ? {$_.CommandLine -match 'laya_overnight'}).Count
 $n=0; if(Test-Path eval\outputs\raw\laya__opendecisions.jsonl){$n=(Get-Content eval\outputs\raw\laya__opendecisions.jsonl).Count}
 $cur=(Select-String -Path eval\outputs\laya_overnight_status.md -Pattern '^Current:' | select -First 1).Line
 $raw=(ls eval\outputs\raw\laya__* | % {$_.Name}) -join ','
 $ck=(ls checkpoints -Directory | % {$_.Name}) -join ','
 Add-Content $log "- $(Get-Date -f 'yyyy-MM-dd HH:mm') IST: overnight_procs=$p OD=$n/2352 | $cur | raw=$raw | ckpt=$ck"
 if($p -eq 0){Add-Content $log "- overnight process exited; watchdog stopping"; break}
 Start-Sleep 300 }
