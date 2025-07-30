# Introduction

### Step 1 :  Setup Certificate Authority Server
```
ansible-playbook -i inventory.yml SSL/certificate_authority_server/setup_ca.yml -e ca_action=install
```
Then you can either deploy the confluent kafka platform or the apache kafka 
