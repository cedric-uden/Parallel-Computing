#!/bin/bash

# automate testing of different runtime configurations
# view `test_setup.png` to view the screen layout


DEFAULT_TIMEOUT=0.3
PROCESSING_TIMEOUT=1.2

###################################

copy_and_paste() {
    xdotool key Ctrl+c
    sleep $DEFAULT_TIMEOUT
    xdotool key Super_L+Right
    sleep $DEFAULT_TIMEOUT
    xdotool key Ctrl+v
}

double_click() {
    xdotool click 1
    sleep $DEFAULT_TIMEOUT
    xdotool click 1
}

next_cell() {
    xdotool key Tab
}

next_row() {
    xdotool key Return
}

###################################


xdotool mousemove 5987 456
sleep $DEFAULT_TIMEOUT

double_click
sleep ${PROCESSING_TIMEOUT}



xdotool mousemove 5180 1855
sleep $DEFAULT_TIMEOUT

double_click
sleep $DEFAULT_TIMEOUT

copy_and_paste
sleep $DEFAULT_TIMEOUT

next_cell
sleep $DEFAULT_TIMEOUT



xdotool mousemove 5180 1900
sleep $DEFAULT_TIMEOUT

double_click
sleep $DEFAULT_TIMEOUT

copy_and_paste
sleep $DEFAULT_TIMEOUT

next_cell
# next_row
sleep $DEFAULT_TIMEOUT



xdotool mousemove 5100 1945
sleep $DEFAULT_TIMEOUT

double_click
sleep $DEFAULT_TIMEOUT

copy_and_paste
sleep $DEFAULT_TIMEOUT

# next_cell
next_row
sleep $DEFAULT_TIMEOUT




# xdotool mousemove 5020 1980
# sleep $DEFAULT_TIMEOUT

# double_click
# sleep $DEFAULT_TIMEOUT

# copy_and_paste
# sleep $DEFAULT_TIMEOUT

# next_row
