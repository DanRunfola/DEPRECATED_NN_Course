#!/usr/bin/env bash

apt-get install -y python3 python3-pip python3-dev

pip3 install -r /autograder/source/requirements.txt

eval `ssh-agent -s`
chmod 600 /autograder/source/id_rsa
ssh-add /autograder/source/id_rsa

cd /autograder
git clone git@github.com:DanRunfola/D442.git