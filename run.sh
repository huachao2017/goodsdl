#/bin/bash
runId=`ps -ef | grep python3 | grep -w 80 | awk 'BEGIN{FS=" "}{print $2}'`
kill -9 $runId
nohup python3 manage.py runserver 0.0.0.0:80 &
