# Build Kafka Streaming Cluster with SSL/TLS using ansible for Confluent and Apache Kafka platforms




### Step 1 :  Setup Certificate Authority Server


```
ansible-playbook -i inventory.yml SSL/certificate_authority_server/setup_ca.yml -e ca_action=install

```

Then you can either deploy the confluent kafka platform or the apache kafka 

### Step 2 : Install Kafka 

####  Confluent Kafka Ansible Deployment - WITH TLS/SSL 
* If you prefer using confluent kafka then execute the following command to install the application : 

1. Create the CSRs and Keystore and get the CSRs signed by the intermediate CA




2. Deploy Confluent with the SSL/TLS 

```
ansible-playbook -i inventory.yml SSL/confluent_kafka_community/deploy_confluent_platform-ssl.yml -e action=install

```

3. Verify the setup - Create a topic and checking the status of different services





####  Apache Kafka  Ansible Deployment - WITH TLS/SSL

* If you prefer using apache kafka  then execute the following command to install the application : 


1. Create the CSRs and Keystore and get the CSRs signed by the intermediate CA



2. Deploy Apache with the SSL/TLS 


```
ansible-playbook -i inventory.yml SSL/apache_kafka/deploy_apache_kafka-ssl.yml -e action=install
```

3. Verify the setup - Create a topic and checking the status of different services


## Bonus - NO SSL Deployment 

### Confluent Kafka Ansible Deployment - WITHOUT SSL



```
ansible-playbook -i inventory.yml NO-SSL/confluent_kafka_community/deploy_confluent_platform.yml -e action=install

```


### Apache Kafka Ansible Deployment - WITHOUT SSL



```
ansible-playbook -i inventory.yml NO-SSL/apache_kafka/deploy_apache_kafka.yml -e action=install

```
