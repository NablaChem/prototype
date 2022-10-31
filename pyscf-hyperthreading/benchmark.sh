run_benchmark() {
    INSTANCE_TYPE=$1
    INSTANCE_CORES=$2
    THREADS_PER_CORE=$3
    KEY_NAME=$4

    echo "Running benchmark for instance type $INSTANCE_TYPE with $THREADS_PER_CORE threads per core and $INSTANCE_CORES cores"
    
    # create instance
    INSTANCE_ID=$(aws ec2 run-instances --instance-type $INSTANCE_TYPE --cpu-options "CoreCount=${INSTANCE_CORES},ThreadsPerCore=${THREADS_PER_CORE}" --key-name $KEY_NAME --image-id ami-09d3b3274b6c5d4aa --region us-east-1 --query 'Instances[0].InstanceId' --output text)
    aws ec2 wait instance-status-ok --instance-ids $INSTANCE_ID --region us-east-1

    # get instance ip
    INSTANCE_IP=$(aws ec2 describe-instances --query 'Reservations[].Instances[].[PublicIpAddress]' --output text --region us-east-1 --instance-ids $INSTANCE_ID)

    # setup SSH access
    SEC_GROUP=$(aws ec2 describe-instance-attribute --region us-east-1 --instance-id $INSTANCE_ID --attribute groupSet --query 'Groups[].GroupId' --output text)
    aws ec2 authorize-security-group-ingress --region us-east-1 --group-id $SEC_GROUP --protocol tcp --port 22 --cidr 0.0.0.0/0 &> /dev/null

    SSH_OPTIONS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    ssh $SSH_OPTIONS ec2-user@$INSTANCE_IP -C "pip3 install --no-warn-script-location pyscf leruli" > /dev/null
    ssh $SSH_OPTIONS ec2-user@$INSTANCE_IP -C "sudo mkfs -t ext4 /dev/nvme1n1; sudo mkdir /scratch; sudo mount /dev/nvme1n1 /scratch; sudo chmod 777 /scratch" > /dev/null
    ssh $SSH_OPTIONS ec2-user@$INSTANCE_IP -C "wget -q https://raw.githubusercontent.com/NablaChem/prototype/main/pyscf-hyperthreading/worker.py -O /scratch/worker.py"

    for molname in benzene naphthalene anthracene; do
        for method in HF PBE CCSD; do 
            ssh $SSH_OPTIONS ec2-user@$INSTANCE_IP -C "cd /scratch; PYSCF_TMPDIR=/scratch python3 worker.py $molname $method"
            ssh $SSH_OPTIONS ec2-user@$INSTANCE_IP -C "cd /scratch; touch tmpfile; rm tmp*"
        done
    done

    # cleanup
    #aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1

}

## run_benchmark c5ad.xlarge 2 1 guido-cli
## run_benchmark c5ad.2xlarge 4 1 guido-cli
## run_benchmark c5ad.4xlarge 8 1 guido-cli
## run_benchmark c5ad.8xlarge 16 1 guido-cli
## run_benchmark c5ad.16xlarge 32 1 guido-cli

# run_benchmark c5ad.xlarge 2 2 guido-cli
# run_benchmark c5ad.2xlarge 4 2 guido-cli
## run_benchmark c5ad.4xlarge 8 2 guido-cli
## run_benchmark c5ad.8xlarge 16 2 guido-cli

