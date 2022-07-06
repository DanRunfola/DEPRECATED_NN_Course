#!/usr/bin/env bash

apt-get install -y python3 python3-pip python3-dev
python3 -m pip install --upgrade pip
pip3 install -r /autograder/source/requirements.txt

eval `ssh-agent -s`
chmod 600 /autograder/source/id_rsa
ssh-add /autograder/source/id_rsa
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts