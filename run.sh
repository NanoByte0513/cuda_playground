set -e

pushd ./
rm -rf ./build_test
mkdir build_test && cd build_test
cmake -DCMAKE_BUILD_TYPE="Debug" .. 
make -j8
popd

# pushd ./
# cd build_test
# make -j8
# popd

./bin/test_cuda
