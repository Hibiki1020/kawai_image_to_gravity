#!/bin/bash

image_name="image_to_gravity"
docker build . \
	-t $image_name:nvidia_docker1 \
	--build-arg CACHEBUST=$(date +%s)
