#!/usr/bin/env bash

./build.sh

docker save joeranbosma/dragon_submission:latest | gzip -c > dragon_submission.tar.gz
