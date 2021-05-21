# python ./aws_launcher.py \
#     --ssh_key_file=../east.pem \
#     --instances= \
#     --regions=us-east-1 \
#     --prepare_cmd="cd ~/CryptGPU; git checkout master; git pull; python3 setup.py install" \
#     --aux_files=benchmark.py,network.py \
#     launcher.py \

python ./aws_launcher.py \
    --ssh_key_file=../east.pem \
    --instances= #put aws instance ids here \
    --regions=us-east-1 \
    --prepare_cmd="" \
    --aux_files=benchmark.py,network.py\
    launcher.py \