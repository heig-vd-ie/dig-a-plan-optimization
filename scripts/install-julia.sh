#!/bin/bash
mkdir DownloadsTemporary23909
cd DownloadsTemporary23909
sudo apt install curl
curl -L -O https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.0-linux-x86_64.tar.gz
tar -xvzf julia-1.11.0-linux-x86_64.tar.gz
sudo mv julia-1.11.0 /opt/
sudo ln -s /opt/julia-1.11.0/bin/julia /usr/local/bin/julia
cd ..
rm -rf DownloadsTemporary23909
julia --version
echo "Julia installation completed successfully."