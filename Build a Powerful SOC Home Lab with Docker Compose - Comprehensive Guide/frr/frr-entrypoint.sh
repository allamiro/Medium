#!/bin/bash
# Clean up any leftover watchfrr temp files
rm -rf /var/tmp/frr/watchfrr*

# Start FRR
/usr/lib/frr/frrinit.sh start
