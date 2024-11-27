# Build Kafka Streaming Cluster with SSL/TLS using ansible (Confluent-Kafka) and Apache Kafka




### Step 1 :  Setup Certificate Authority Server


```
ansible-playbook -i inventory.yml SSL/certificate_authority_server/setup_ca.yml -e ca_action=install

```

Then you can either deploy the confluent kafka platform or the apache kafka 

### Step 2 : Install Kafka 

####  Confluent Kafka Ansible Deployment - WITH TLS/SSL 

* If you prefer using confluent kafka then execute the following command to install the application : 

```
ansible-playbook -i inventory.yml SSL/confluent_kafka_community/deploy_confluent_platform-ssl.yml -e action=install

```


####  Apache Kafka  Ansible Deployment - WITH TLS/SSL

* If you prefer using apache kafka  then execute the following command to install the application : 

```
ansible-playbook -i inventory.yml SSL/apache_kafka/deploy_apache_kafka-ssl.yml -e action=install
```



## Bonus - NO SSL Deployment 

### Confluent Kafka Ansible Deployment - WITHOUT SSL



```
ansible-playbook -i inventory.yml NO-SSL/confluent_kafka_community/deploy_confluent_platform.yml -e action=install

```


### Apache Kafka Ansible Deployment - WITHOUT SSL



```
ansible-playbook -i inventory.yml NO-SSL/apache_kafka/deploy_apache_kafka.yml -e action=install

```