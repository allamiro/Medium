# Introduction

### Step 1 :  Setup Certificate Authority Server
```
ansible-playbook -i inventory.yml SSL/certificate_authority_server/setup_ca.yml -e ca_action=install
```
Then you can either deploy the confluent kafka platform or the apache kafka 


### Step 2 : Prepare the Servers 

1. Install the required packages

2. Update the hosts file 

3. Copy the Certificate Server file to all Systems and update the trust 


### Step 3 :  Deploy Kuberentes on the first server (kube1)



### Step 4 :  Deploy Kubernetes on the second server (kube2)



### Step 5 : Generate the Certificate Server Request and Get them signed 


### Step 6 :  Deploy MinIO on the first server (kube1)


### Step 7 : Deploy MinIO on the second server (kube2)

