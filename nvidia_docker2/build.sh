#!/bin/bash

image_name="image_to_gravity"
docker build . \
	-t $image_name:nvidia_docker2 \
	--build-arg CACHEBUST=$(date +%s)
