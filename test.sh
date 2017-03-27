#########################################################################
# File Name: test.sh
# Author: qiao_yuchen
# mail: qiaoyc14@mails.tsinghua.edu.cn
# Created Time: Mon 09 Jan 2017 12:18:33 AM JST
#########################################################################
#!/bin/bash
rm ./model.1itr.bin
rm ./temp.txt
./n3lp -d ../n3lp_data/1k/ -i 512 -h 512 -m 128 -n 1 -v 5
../play_eigen/play4/mainTest ./model.1itr.bin ./temp.txt
