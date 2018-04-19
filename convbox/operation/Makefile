
TF_INC=`python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`

so:convbox_match.cc
	g++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared convbox_match.cc box.cpp -o convbox_match.so -fPIC -I./ -I${TF_INC} -O2

clean:
	rm convbox_match.so
