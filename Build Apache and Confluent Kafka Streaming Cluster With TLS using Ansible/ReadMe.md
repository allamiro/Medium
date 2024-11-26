# Build Kafka Streaming Cluster with TLS using ansible (Confluent-Kafka) and Apache Kafka

# Setup Certificate Authority Server

``` cd certificate_authority_server```

```ansible-playbook -i inventory.yml setup_ca.yml```

### Confluent Kafka Ansible Deployment 

```cd apache-kafka```

```ansible-playbook -i inventory.yml deploy_confluent_platform.yml```


### Apache Kafka Ansible Deployment

```cd confluent-kafka```

```ansible-playbook -i inventory.yml deploy_apache_kafka.yml```