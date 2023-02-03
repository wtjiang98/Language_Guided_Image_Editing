#!/usr/bin/expect -f

set PROJECT_PATH "/mnt/data1/jwt/LGIE-Django"
set PYTHON_PATH "/home/jwt/anaconda3/bin/python3"
set USER_NAME "jwt"
set PASSWORD "jwt"

set input_path [lindex $argv 0]
set request [lindex $argv 1]
set output_path [lindex $argv 2]

puts $input_path
puts $request
puts $output_path
spawn su - $USER_NAME -s $PROJECT_PATH/LGIE/test_single_GIER.sh $input_path $request $output_path
expect "Password:"
send "$PASSWORD\r"
interact

