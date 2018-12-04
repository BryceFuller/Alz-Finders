## declare an array variable of all ips
declare -a arr=("element1" "element2" "element3")

## now loop through the ips running the script
for i in "${arr[@]}"
do
   scp -i <keypair> myfile.txt ubuntu@ec2-$i.us-west-2.compute.amazonaws.com/home/ubuntu/myfile.txt
   ssh -i "ds-test-key.pem" ubuntu@ec2-$i.us-west-2.compute.amazonaws.com screen -d -m python thing.py
done
