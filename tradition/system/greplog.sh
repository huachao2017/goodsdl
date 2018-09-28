#!/bin/bash


regx=''
file='access'
param=''
showserver=0
showdate=0
startdate=`date +"%F"`
enddate=`date +"%F"`
onlydate=''
cutLog=0

while getopts "aep:x:vS:E:O:dc" arg
do
  case $arg in
     a)
         #echo grep acess.log
         ;;
     e)
         #echo grep error.log
         file='error'
         ;;
     x)
         #echo 'grep s param is:'"$OPTARG"
         param=$OPTARG
         ;;
     p)
         #echo grep pattern is: "$OPTARG"
         regx=$OPTARG
         ;;
     v)
         #echo this will show server name
         showserver=1
         ;;
     d)
         #echo this will show date
         showdate=1
         ;;
     S)
         #echo the start date is "$OPTARG"
         startdate=`date +"%F" -d "$OPTARG"`
         ;;
     E)
         #echo the end date is "$OPTARG"
         enddate=`date +"%F" -d "$OPTARG"`
         ;;
     O)
         #echo the only date is "$OPTARG"
         onlydate=`date +"%F" -d "$OPTARG"`
         ;;
     c)
         #echo 处理日志 "$OPTARG"
         cutLog=1
         ;;
     ?)
         #echo "现在还不支持" [$midi] "这个参数！"
         ;;
  esac
done


pdate() {

  local today=`date +"%Y%m%d"`
  local result=''
  if [ $1 == $today ]
  then
    echo ''
  else
    echo "_$1"
  fi

}

#只统计这一天
if [ "x$onlydate" != "x" ]
then
  startdate=$onlydate
  enddate=$onlydate
fi

#开始和结束的时间数值i.e. 20130504
startnum=`date +"%Y%m%d" -d "$startdate"`
endnum=`date +"%Y%m%d" -d "$enddate"`

while(($startnum<=$endnum));
do
  filedate=`pdate $startnum`
  #最后的文件名
  filename=$file$filedate
  
  if [ $showdate -eq 1 ]
  then
    echo "======================================================== $startdate ========================================================="
  fi

  for j in {1..4};
  do
     if [ $showserver -eq 1 ]
     then
       echo "----------------============= NGINX $j ===============---------------------"
     fi

	 if [ $cutLog -eq 1 ]
	 then
	     # 处理成用空格分开的标准格式
		 ssh "nginx$j" "grep '"$regx"' -P /data/work/nginx/logs/$filename.log $param" | grep -P ':[0-9]+(-[.0-9]+){4}-(-|[.0-9]+)' \
		     | sed -r -e 's/ - - \[([0-9]+)\/([a-zA-Z]+)\/([0-9]+):([0-9]+:[0-9]+:[0-9]+) \+[0-9]+\]/ \3-\2-\1 \4 /' \
			 -e 's/Jan/01/' -e 's/Feb/02/' -e 's/Mar/03/' -e 's/Apr/04/' -e 's/May/05/' -e 's/Jun/06/' \
			 -e 's/Jul/07/' -e 's/Aug/08/' -e 's/Sep/09/' -e 's/Oct/10/' -e 's/Nov/11/' -e 's/Dec/12/' \
			 -e 's/"//' -e 's/".*$//g' -e 's/:([0-9]+)-([.0-9]+)-([.0-9]+)-([.0-9]+)-([.0-9]+)-(-|[.0-9]+)/ \1 \2 \3 \4 \5 \6/'
	 else
		 ssh "nginx$j" "grep '"$regx"' -P /data/work/nginx/logs/$filename.log $param"
	 fi
  done


  #next day
  startdate=`date +"%F" -d "1 day $startdate"`
  startnum=`date +"%Y%m%d" -d "$startdate"`

done

















