set -e

pushd ./
rm -rf ./build_test
mkdir build_test && cd build_test
cmake .. 
make -j8
popd

./bin/test_cuda
