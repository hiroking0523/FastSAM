xhost +
sudo docker run -it --rm --runtime nvidia --shm-size=1g -v $(pwd)/home:/home -e DISPLAY=:0 --network host fast_sam:latest