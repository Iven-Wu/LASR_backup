module load gcc
module load cmake

conda env create -f lasr.yml
conda activate lasr

pip install scikit-image

pip install geomloss

pip install -e third_party/softras/

git clone --recursive https://github.com/hjwdzh/Manifold.git; cd Manifold; mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release;make -j8; cd ../../

