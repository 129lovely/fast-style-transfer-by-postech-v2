#!/bin/bash

while true; do
    if [ -f print_out_container.sh ]; then
        bash ./print_out_container.sh && rm -rf print_out_container.sh
    else
        echo "waiting for printing command..."
    fi
sleep 1
done