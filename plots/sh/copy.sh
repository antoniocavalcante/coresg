#!/bin/bash

AWS_KEY="/home/toni/AWS/vm-amazon.pem"
USER="ubuntu"

kmaxdim() {
    SERVER="0.0.0.0"

    scp -i ${AWS_KEY} ${USER}@${SERVER}:/home/ubuntu/hdbscan_python/handl*.results ../results/kmaxdim/
}

performance(){
    SERVER_KNN="0.0.0.0"
    SERVER_RNG="0.0.0.0"
    SERVER_KNN_INC="0.0.0.0"

    # Copy KNN results
    for SERVER in ${SERVER_RNG} ${SERVER_KNN} ${SERVER_KNN_INC};
    do
        scp -i ${AWS_KEY} ${USER}@${SERVER}:/home/ubuntu/hdbscan_python/*.results ../results/
    done
}

all() {
    SERVER="0.0.0.0"

    scp -i ${AWS_KEY} ${USER}@${SERVER}:/home/ubuntu/hdbscan_python/real*.results ../results/
}


all
