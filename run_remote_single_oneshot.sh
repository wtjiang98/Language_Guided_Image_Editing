#!/usr/bin/expect -f
set input_path [lindex $argv 0]
set request [lindex $argv 1]
set output_path [lindex $argv 2]

puts $input_path
puts $request
puts $output_path
spawn su - jwt -s /mnt/data1/jwt/LGIE/test_single.sh $input_path $request $output_path
expect "Password:"
send "jwt\r"
interact

