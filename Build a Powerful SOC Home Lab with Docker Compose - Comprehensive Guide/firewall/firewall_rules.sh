#!/bin/sh

# Flush existing rules
iptables -F
iptables -t nat -F

# Enable NAT (Masquerading) for traffic from DMZ to External network
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# Allow forwarding of established connections
iptables -A FORWARD -m state --state RELATED,ESTABLISHED -j ACCEPT

# Block all incoming traffic from External to Management and Internal
iptables -A FORWARD -i eth0 -o eth2 -j DROP  # Block External to Management
iptables -A FORWARD -i eth0 -o eth1 -j DROP  # Block External to Internal

# Allow HTTP and HTTPS traffic from External to DMZ
iptables -A FORWARD -i eth0 -o eth1 -p tcp --dport 80 -j ACCEPT    # Allow HTTP to DMZ
iptables -A FORWARD -i eth0 -o eth1 -p tcp --dport 443 -j ACCEPT   # Allow HTTPS to DMZ

# Allow essential traffic from Management to Internal
iptables -A FORWARD -i eth2 -o eth1 -p tcp --dport 22 -j ACCEPT     # SSH
iptables -A FORWARD -i eth2 -o eth1 -p tcp --dport 443 -j ACCEPT    # HTTPS
iptables -A FORWARD -i eth2 -o eth1 -p udp --dport 161 -j ACCEPT    # SNMP
iptables -A FORWARD -i eth2 -o eth1 -p udp --dport 123 -j ACCEPT    # NTP
iptables -A FORWARD -i eth2 -o eth1 -p tcp --dport 3389 -j ACCEPT   # RDP
iptables -A FORWARD -i eth2 -o eth1 -p udp --dport 67:68 -j ACCEPT  # DHCP (if needed)

# Permit DNS queries from Management and Internal to DMZ or External DNS servers
iptables -A FORWARD -i eth1 -o eth0 -p udp --dport 53 -j ACCEPT     # DNS UDP
iptables -A FORWARD -i eth1 -o eth0 -p tcp --dport 53 -j ACCEPT     # DNS TCP
iptables -A FORWARD -i eth2 -o eth0 -p udp --dport 53 -j ACCEPT     # DNS UDP
iptables -A FORWARD -i eth2 -o eth0 -p tcp --dport 53 -j ACCEPT     # DNS TCP

# Permit mail traffic from Internal and Management to DMZ Mail Server
iptables -A FORWARD -i eth1 -o eth0 -p tcp --dport 25 -j ACCEPT     # SMTP
iptables -A FORWARD -i eth1 -o eth0 -p tcp --dport 110 -j ACCEPT    # POP3
iptables -A FORWARD -i eth1 -o eth0 -p tcp --dport 143 -j ACCEPT    # IMAP
iptables -A FORWARD -i eth2 -o eth0 -p tcp --dport 25 -j ACCEPT     # SMTP
iptables -A FORWARD -i eth2 -o eth0 -p tcp --dport 110 -j ACCEPT    # POP3
iptables -A FORWARD -i eth2 -o eth0 -p tcp --dport 143 -j ACCEPT    # IMAP

# Allow limited traffic from Internal to DMZ
iptables -A FORWARD -i eth1 -o eth0 -p tcp --dport 80 -j ACCEPT     # HTTP
iptables -A FORWARD -i eth1 -o eth0 -p tcp --dport 443 -j ACCEPT    # HTTPS

# Allow limited traffic from DMZ to Internal
iptables -A FORWARD -i eth0 -o eth1 -p tcp --dport 80 -j ACCEPT     # HTTP
iptables -A FORWARD -i eth0 -o eth1 -p tcp --dport 443 -j ACCEPT    # HTTPS

# Drop all other forwarding traffic
iptables -A FORWARD -j DROP
