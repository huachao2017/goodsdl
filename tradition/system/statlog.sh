#!/bin/bash
# 6为res code，7为res time，11为request
# awk只能通过一维数组模拟二维数组

outfile="stat_$2_$1.txt"
greplog.sh -O $1 -p $2 -c | awk -F "[ ?]+" '
{
if(($11,$6) in arr){
arr[$11,$6]+=1;
}else{
arr[$11,$6]=1;
}
if(($11,1) in arr){
arr[$11,1]+=$7;
}else{
arr[$11,1]=$7;
}
}
END{
for(item in arr){
split(item ,arr2,SUBSEP);
if(arr2[2] == 1){
time[arr2[1]]=arr[arr2[1],arr2[2]];
}else{
if(arr2[1] in num){
num[arr2[1]]+=arr[arr2[1],arr2[2]];
other_num[arr2[1]] = sprintf("%s %d:%d",other_num[arr2[1]],arr2[2],arr[arr2[1],arr2[2]]);
}else{
num[arr2[1]]=arr[arr2[1],arr2[2]];
other_num[arr2[1]] = sprintf("%d:%d",arr2[2],arr[arr2[1],arr2[2]]);
}
}
}
for(item in time){
print num[item],time[item]/num[item],item,other_num[item];
}
}'   | sort -rn > $outfile
