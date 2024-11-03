#!/bin/bash
# Start FRR services
chown -R frr:frr /etc/frr
/usr/lib/frr/frrinit.sh start
# Keep the container running
tail -f /dev/null
